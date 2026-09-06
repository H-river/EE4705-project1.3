#!/usr/bin/env python
# Owner: backbone (ALL)
"""Multi-camera synchronization check (diagnostic scene with a falling ball).

Runs a deterministic six-second episode on ``assets/scene_cam_sync.xml``:

* base and waist targets held fixed;
* the right arm reaches toward a declared tabletop target and returns;
* a magenta diagnostic ball is released from ~0.30 m above the table top
  at t = 0 (it is NOT a manipulable object and cannot be attached).

Control and capture share ONE simulation-step scheduler: every requested
capture time is aligned to a physics step, and at that step all three
cameras are captured atomically through ``RobotEnv.get_obs_multi`` (no
stepping between cameras, no controller update inside the capture).

Capture schedule (nominal times):
1. overview rows   : 0.00, 0.25, ..., 5.75 s (24 rows)            -> contact_sheet.png
2. video           : 0.0, 0.1, ..., 5.9 s (60 frames, 10 fps)     -> sync.mp4 (GIF fallback)
3. dense burst     : 0.00 .. 0.60 s every 0.01 s (100 Hz, 61 frames) covering the fall/impact

Outputs under runs/cam_sync/<run_id>/ (see README in that directory).
Exit code 0 on completion, 1 when any capture failed or a batch was found
non-synchronized.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.env import RobotEnv  # noqa: E402
from core.g1 import CAMERAS, WAIST_REST  # noqa: E402
from core.rendering import mujoco  # noqa: E402
from core.types import Observation, SceneConfig, SceneObjectSpec  # noqa: E402
from core.world import SCENE_CAM_SYNC_XML, SimWorld  # noqa: E402

TABLE_TOP_Z = 0.85
EPISODE_S = 6.0
REACH_TARGET = np.array([0.36, -0.20, TABLE_TOP_Z + 0.05])  # declared tabletop target (world frame)
REACH_AT_S = 0.5  # command the reach at this time
RETURN_AT_S = 3.0  # command the return (to the initial EE position) at this time
BALL_START = np.array([0.34, -0.12, TABLE_TOP_Z + 0.30])  # ball CENTER 0.30 m above the table top
BALL_RADIUS = 0.03

OVERVIEW_TIMES = [0.25 * i for i in range(24)]
VIDEO_TIMES = [0.1 * i for i in range(60)]
BURST_TIMES = [0.01 * i for i in range(61)]  # 100 Hz over the first 0.60 s (fall ~0.25 s + impact/bounce)

DEPTH_VMIN, DEPTH_VMAX = 0.2, 2.5  # m, fixed color scale of every depth preview (documented)

# magenta ball colour segmentation (RGB uint8): strong red and blue, weak green
BALL_RGB_RULE = "R >= 120 and B >= 90 and G <= 80 and R - G >= 70"
MIN_BALL_PIXELS = 4


def detect_ball(rgb: np.ndarray) -> tuple[float, float, int]:
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    mask = (r >= 120) & (b >= 90) & (g <= 80) & ((r - g) >= 70)
    n = int(mask.sum())
    if n < MIN_BALL_PIXELS:
        return math.nan, math.nan, n
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean()), n


def project(obs: Observation, p_world: np.ndarray) -> tuple[float, float, float]:
    """(u, v, depth) of a world point in the observation's camera."""
    t_cam_world = np.linalg.inv(obs.t_world_camera)
    pc = t_cam_world @ np.array([p_world[0], p_world[1], p_world[2], 1.0])
    if pc[2] <= 1e-6:
        return math.nan, math.nan, float(pc[2])
    k = obs.intrinsics
    return float(k[0, 0] * pc[0] / pc[2] + k[0, 2]), float(k[1, 1] * pc[1] / pc[2] + k[1, 2]), float(pc[2])


def depth_preview(depth: np.ndarray) -> np.ndarray:
    """Fixed-scale depth colouring: DEPTH_VMIN (near, yellow) .. DEPTH_VMAX
    (far, dark blue); NaN/invalid -> black."""
    import matplotlib
    cmap = matplotlib.colormaps["viridis_r"]
    d = np.clip((depth - DEPTH_VMIN) / (DEPTH_VMAX - DEPTH_VMIN), 0.0, 1.0)
    img = (cmap(np.nan_to_num(d, nan=0.0))[..., :3] * 255).astype(np.uint8)
    img[np.isnan(depth)] = 0
    return img


_FONT = None


def font(size: int = 18):
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.load_default(size=size)
        except TypeError:  # older Pillow
            _FONT = ImageFont.load_default()
    return _FONT


def overlay(rgb: np.ndarray, lines: list[str]) -> Image.Image:
    im = Image.fromarray(rgb).copy()
    dr = ImageDraw.Draw(im)
    y = 4
    for line in lines:
        dr.rectangle([2, y - 1, 8 + 9 * len(line), y + 20], fill=(0, 0, 0))
        dr.text((5, y), line, fill=(255, 255, 0), font=font())
        y += 22
    return im


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="G1 multi-camera synchronization check")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/cam_sync"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / run_id
    n = 1
    while out.exists():
        n += 1
        out = args.out / f"{run_id}_{n}"
    (out / "frames").mkdir(parents=True)
    (out / "depth").mkdir()
    (out / "video_frames").mkdir()

    world = SimWorld(SCENE_CAM_SYNC_XML)
    env = RobotEnv(world)
    dt = world.timestep
    try:
        return _run(world, env, dt, out, run_id, make_video=not args.no_video)
    finally:
        world.close()


def _run(world: SimWorld, env: RobotEnv, dt: float, out: pathlib.Path, run_id: str, make_video: bool) -> int:
    scene = SceneConfig(seed=0, objects=[SceneObjectSpec("stone", (0.40, -0.15, 0.88)),
                                         SceneObjectSpec("cube", (0.40, 0.15, 0.88)),
                                         SceneObjectSpec("bottle", (0.55, -0.35, 0.915))])
    world.reset(scene)
    world.set_waist_target(WAIST_REST["waist_yaw_joint"], WAIST_REST["waist_roll_joint"], WAIST_REST["waist_pitch_joint"])
    # settle objects (and the arm) BEFORE the episode clock starts, then re-zero time
    world.step(int(round(1.0 / dt)))
    world.data.time = 0.0
    # place the ball (free body; not in MANIPULABLE_BODIES) and release it at t = 0
    adr = world.model.joint("diag_ball_free").qposadr[0]
    dof = world.model.joint("diag_ball_free").dofadr[0]
    world.data.qpos[adr:adr + 3] = BALL_START
    world.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    world.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(world.model, world.data)
    initial_ee = world.ee_pos().copy()

    # ---- common step schedule: step index -> set of capture kinds
    def to_step(t: float) -> int:
        return int(round(t / dt))

    schedule: dict[int, set[str]] = {}
    nominal: dict[tuple[int, str], float] = {}
    for kind, times in (("overview", OVERVIEW_TIMES), ("video", VIDEO_TIMES), ("burst", BURST_TIMES)):
        for t in times:
            k = to_step(t)
            schedule.setdefault(k, set()).add(kind)
            nominal[(k, kind)] = t
    control_events = {to_step(REACH_AT_S): "reach", to_step(RETURN_AT_S): "return"}
    last_step = to_step(EPISODE_S)

    rows: list[dict] = []
    batches: list[dict] = []
    contact_rows: list[tuple[float, dict[str, np.ndarray]]] = []
    video_frames: list[tuple[float, dict[str, np.ndarray]]] = []
    calib: dict[str, dict] = {}
    capture_failures: list[str] = []
    max_state_change = 0.0
    max_time_change = 0.0
    render_wall = 0.0
    n_captures = 0
    print(f"Episode: {EPISODE_S:.1f} s at dt={dt} ({last_step} steps); reach {REACH_TARGET.tolist()} at t={REACH_AT_S}s, "
          f"return at t={RETURN_AT_S}s; ball released from z={BALL_START[2]:.2f} m (table top {TABLE_TOP_Z})")
    print(f"Capture steps: {len(schedule)} distinct (overview {len(OVERVIEW_TIMES)}, video {len(VIDEO_TIMES)}, burst {len(BURST_TIMES)})")

    t_wall0 = time.time()
    for k in range(last_step + 1):
        if k > 0:
            world.step(1)
        # control events (targets only; motion happens in the stepping)
        ev = control_events.get(k)
        if ev == "reach":
            world.set_arm_target(REACH_TARGET)
        elif ev == "return":
            world.set_arm_target(initial_ee)
        kinds = schedule.get(k)
        if not kinds:
            continue
        # ---- atomic multi-camera capture at this step
        qpos_before = world.data.qpos.copy()
        qvel_before = world.data.qvel.copy()
        time_before = world.sim_time
        t0 = time.perf_counter()
        try:
            obs = env.get_obs_multi(list(CAMERAS))
        except Exception as exc:  # noqa: BLE001
            capture_failures.append(f"step {k}: {type(exc).__name__}: {exc}")
            continue
        render_wall += time.perf_counter() - t0
        n_captures += 1
        max_state_change = max(max_state_change, float(np.max(np.abs(world.data.qpos - qpos_before))),
                               float(np.max(np.abs(world.data.qvel - qvel_before))))
        max_time_change = max(max_time_change, abs(world.sim_time - time_before))
        times = [o.sim_time for o in obs.values()]
        ids = [o.capture_id for o in obs.values()]
        ball = world.body_pos("diag_ball")
        ee = world.ee_pos()
        batch = {"step": k, "actual_t": world.sim_time, "kinds": sorted(kinds),
                 "max_inter_camera_dt": max(times) - min(times), "capture_ids": ids,
                 "ball_pos": ball.tolist(), "ee_pos": ee.tolist()}
        for kind in kinds:
            batch[f"nominal_{kind}"] = nominal[(k, kind)]
            batch[f"sched_err_{kind}"] = world.sim_time - nominal[(k, kind)]
        batches.append(batch)
        per_cam = {}
        for cam, o in obs.items():
            u, v, area = detect_ball(o.rgb)
            pu, pv, pz = project(o, ball)
            in_view = (not math.isnan(pu)) and 0 <= pu < o.rgb.shape[1] and 0 <= pv < o.rgb.shape[0]
            per_cam[cam] = {"u": u, "v": v, "area": area, "proj_u": pu, "proj_v": pv, "proj_depth": pz, "in_view": in_view,
                            "reproj_err_px": math.hypot(u - pu, v - pv) if (not math.isnan(u) and not math.isnan(pu)) else math.nan}
            calib[o.capture_id] = {"camera": cam, "sim_time": o.sim_time, "frame_id": o.frame_id,
                                   "intrinsics": o.intrinsics.tolist(), "t_world_camera": o.t_world_camera.tolist()}
            rows.append({"capture_id": o.capture_id, "camera": cam, "step": k, "actual_t": o.sim_time,
                         "kinds": "+".join(sorted(kinds)), **{f"nominal_{kd}": nominal[(k, kd)] for kd in kinds},
                         "ball_u": u, "ball_v": v, "ball_area_px": area, "ball_visible": not math.isnan(u),
                         "ball_proj_u": pu, "ball_proj_v": pv, "ball_in_view": in_view,
                         "reproj_err_px": per_cam[cam]["reproj_err_px"],
                         "ball_x": ball[0], "ball_y": ball[1], "ball_z": ball[2], "ee_x": ee[0], "ee_y": ee[1], "ee_z": ee[2]})
        batch["per_camera"] = per_cam
        # ---- artifacts
        if "overview" in kinds or "burst" in kinds:
            for cam, o in obs.items():
                stem = o.capture_id
                lines = [f"t={o.sim_time:.3f}s  {o.capture_id}"]
                overlay(o.rgb, lines).save(out / "frames" / f"{stem}_rgb.png")
                if "overview" in kinds:
                    Image.fromarray(depth_preview(o.depth)).save(out / "frames" / f"{stem}_depth.png")
                    np.savez_compressed(out / "depth" / f"{stem}_depth.npz", depth=o.depth, intrinsics=o.intrinsics,
                                        t_world_camera=o.t_world_camera, sim_time=o.sim_time)
        if "overview" in kinds:
            contact_rows.append((world.sim_time, {c: o.rgb for c, o in obs.items()}))
        if "video" in kinds:
            video_frames.append((world.sim_time, {c: (o.rgb, o.capture_id) for c, o in obs.items()}))
    wall = time.time() - t_wall0

    # ---- contact sheet: 24 rows x 3 cameras, thumbnails 320x240
    tw, th, label_w = 320, 240, 110
    sheet = Image.new("RGB", (label_w + 3 * tw, len(contact_rows) * th + 30), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for j, cam in enumerate(CAMERAS):
        dr.text((label_w + j * tw + 8, 6), cam, fill=(255, 255, 255), font=font())
    for i, (t, imgs) in enumerate(contact_rows):
        y = 30 + i * th
        dr.text((6, y + th // 2 - 10), f"t={t:.2f} s", fill=(255, 255, 0), font=font())
        for j, cam in enumerate(CAMERAS):
            sheet.paste(Image.fromarray(imgs[cam]).resize((tw, th)), (label_w + j * tw, y))
    sheet.save(out / "contact_sheet.png")

    # ---- video: synchronized triptych, 10 fps
    video_path = None
    video_note = ""
    if make_video:
        for i, (t, frames) in enumerate(video_frames):
            trip = Image.new("RGB", (3 * 640, 480 + 28), (0, 0, 0))
            d2 = ImageDraw.Draw(trip)
            d2.text((8, 5), f"sim t = {t:.3f} s   (10 fps nominal, actual timestamps)", fill=(255, 255, 0), font=font())
            for j, cam in enumerate(CAMERAS):
                rgb, cid = frames[cam]
                trip.paste(overlay(rgb, [f"{cam}  t={t:.3f}s", cid]), (j * 640, 28))
            trip.save(out / "video_frames" / f"frame_{i:04d}.png")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            video_path = out / "sync.mp4"
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(out / "video_frames" / "frame_%04d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(video_path)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0 or not video_path.exists():
                video_note = f"ffmpeg failed ({res.stderr.strip()[:200]}); "
                video_path = None
        if video_path is None:
            video_path = out / "sync.gif"
            frames_gif = [Image.open(out / "video_frames" / f"frame_{i:04d}.png").resize((3 * 320, 254))
                          for i in range(len(video_frames))]
            frames_gif[0].save(video_path, save_all=True, append_images=frames_gif[1:], duration=100, loop=0)
            video_note += "GIF fallback used (no usable ffmpeg)"

    # ---- diagnostics
    fields = list(rows[0].keys()) if rows else []
    with (out / "captures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "calibration.json").write_text(json.dumps(calib, indent=1))
    burst = [b for b in batches if "burst" in b["kinds"]]
    max_inter = max(b["max_inter_camera_dt"] for b in batches)
    sched_err = max(abs(b[k]) for b in batches for k in b if k.startswith("sched_err_"))
    ids_all = [cid for b in batches for cid in b["capture_ids"]]
    visible_during_fall = {cam: sum(1 for b in burst if not math.isnan(b["per_camera"][cam]["u"])) for cam in CAMERAS}
    in_view_during_fall = {cam: sum(1 for b in burst if b["per_camera"][cam]["in_view"]) for cam in CAMERAS}
    reproj = {cam: [b["per_camera"][cam]["reproj_err_px"] for b in batches if not math.isnan(b["per_camera"][cam]["reproj_err_px"])]
              for cam in CAMERAS}
    impact_t = next((b["actual_t"] for b in burst if b["ball_pos"][2] <= TABLE_TOP_Z + BALL_RADIUS + 0.002), None)
    summary = {
        "run_id": run_id, "dt": dt, "episode_s": EPISODE_S, "reach_target": REACH_TARGET.tolist(),
        "ball_start": BALL_START.tolist(), "ball_color_rule": BALL_RGB_RULE, "depth_preview_scale_m": [DEPTH_VMIN, DEPTH_VMAX],
        "n_batches": len(batches), "n_observations": len(rows), "capture_failures": capture_failures,
        "max_inter_camera_timestamp_diff_s": max_inter, "max_scheduling_error_s": sched_err,
        "max_physical_state_change_across_capture": max_state_change, "max_sim_time_change_across_capture": max_time_change,
        "unique_capture_ids": len(set(ids_all)) == len(ids_all),
        "ball_visible_detections_during_fall_burst": visible_during_fall,
        "ball_geometrically_in_view_during_fall_burst": in_view_during_fall,
        "ball_first_table_contact_t": impact_t,
        "ball_reprojection_error_px": {cam: {"n": len(v), "mean": float(np.mean(v)) if v else None, "max": float(np.max(v)) if v else None}
                                       for cam, v in reproj.items()},
        "render_throughput_obs_per_s": n_captures * len(CAMERAS) / render_wall if render_wall > 0 else None,
        "render_wall_s": render_wall, "episode_wall_s": wall,
        "contact_sheet": str(out / "contact_sheet.png"), "video": str(video_path) if video_path else None, "video_note": video_note,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    (out / "README.txt").write_text(
        "frames/<capture_id>_rgb.png   overview + burst RGB frames with sim time and observation id overlaid\n"
        "frames/<capture_id>_depth.png overview depth previews, fixed scale %.1f..%.1f m (viridis_r; black = invalid/NaN)\n"
        "depth/<capture_id>_depth.npz  raw metric depth (float32, NaN invalid) + K + T_world_camera + sim_time (overview rows)\n"
        "calibration.json              per-observation K / T_world_camera / sim_time for EVERY capture\n"
        "captures.csv                  per-observation timing, ball colour-centroid, mask area, projected centre, reprojection error\n"
        "contact_sheet.png             24 rows (t = 0.00 .. 5.75 s) x 3 cameras\n"
        "sync.mp4 / sync.gif           synchronized triptych, 10 fps, 6 s (actual timestamps overlaid)\n"
        "summary.json                  synchronization diagnostics\n" % (DEPTH_VMIN, DEPTH_VMAX))

    print(f"\nBatches: {len(batches)} ({len(rows)} observations); capture failures: {len(capture_failures)}")
    print(f"Max inter-camera timestamp difference (any batch): {max_inter:.3e} s")
    print(f"Max scheduling error |actual - nominal|: {sched_err:.3e} s (all requests aligned to {dt} s steps)")
    print(f"Physical-state change across capture: max |dqpos|,|dqvel| = {max_state_change:.3e}; sim-time change = {max_time_change:.3e} s")
    print(f"All capture IDs unique: {summary['unique_capture_ids']}")
    print("Per-camera ball detections during the 100 Hz fall burst (61 frames): "
          + ", ".join(f"{c}: {visible_during_fall[c]} detected / {in_view_during_fall[c]} geometrically in view" for c in CAMERAS))
    print(f"Ball first table contact at t = {impact_t}")
    for c in CAMERAS:
        e = summary["ball_reprojection_error_px"][c]
        print(f"  {c}: colour-centroid vs projected-centre error: n={e['n']} mean={e['mean']} px max={e['max']} px")
    for b in [x for x in batches if "overview" in x["kinds"]]:
        pc = b["per_camera"]
        print(f"  t={b['actual_t']:.2f}s " + "  ".join(
            f"{c}: ({pc[c]['u']:.0f},{pc[c]['v']:.0f}) area={pc[c]['area']}" if not math.isnan(pc[c]["u"]) else f"{c}: (--,--) area={pc[c]['area']}"
            for c in CAMERAS))
    print(f"Rendering throughput: {summary['render_throughput_obs_per_s']:.1f} observations/s "
          f"({n_captures * len(CAMERAS)} observations in {render_wall:.1f} s render wall; episode wall {wall:.1f} s)")
    print(f"Contact sheet: {out / 'contact_sheet.png'}")
    print(f"Video: {video_path} {video_note}")
    ok = not capture_failures and max_inter == 0.0 and max_state_change == 0.0 and summary["unique_capture_ids"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
