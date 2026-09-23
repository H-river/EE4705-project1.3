"""Render an e2e episode to mp4 from the frames the runner already saved (no live calls).

usage: python scripts/render_episode.py <trial_dir> <out.mp4> [--audit-dir runs/night/audit_a]

<trial_dir> is the directory holding trial_record.json and frames/. Each
head-camera frame becomes one slide: the frames the runner saved, plus every
frame Student A analysed in that episode (A's audit session is found by
matching PNG hashes, and it also keeps SEARCH views and frames the provider
refused). A slide shows A's detections for that exact image and a caption
listing the events since the previous frame (plan, action result,
clarification, verification). This is a slideshow of observations, not
continuous video.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, CAPTION_H = 640, 480, 230
SLIDE_S, QUIET_S, CARD_S = 1.6, 0.8, 3.0
STATUS_COLOR = {"LOCALIZED": (40, 200, 60), "UNLOCALIZED": (240, 180, 0), "AMBIGUOUS": (230, 60, 200)}


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_SMALL, F_MED, F_BIG = font(13), font(15), font(22)


def a_png_sha(rgb: np.ndarray) -> str:
    # Same encoding as StudentAPerception._observe, so the hash matches its audit.
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def episode_audits(audit_dir: pathlib.Path, shas: set[str]) -> list[dict]:
    """All of A's audits from the audit session (one per episode) that saw these frames."""
    sessions: dict[pathlib.Path, int] = {}
    for f in audit_dir.rglob("audit.json"):
        text = f.read_text()
        if any(s in text for s in shas) and json.loads(text).get("image_sha256") in shas:
            sessions[f.parent.parent] = sessions.get(f.parent.parent, 0) + 1
    if not sessions:
        return []
    session = max(sessions, key=sessions.get)
    return [json.loads(f.read_text()) for f in sorted(session.glob("*/audit.json"))]


def event_line(e: dict) -> str:
    t = e["type"]
    if t == "plan":
        s = f"PLAN {e.get('status')}: {' > '.join(e.get('actions') or [])}"
        return s + (f"  ({e['reason']})" if e.get("reason") and not e.get("actions") else "")
    if t == "action":
        ok = "ok" if e.get("success") else f"FAIL {e.get('error')}"
        return f"{e['skill']} {e.get('target') or ''} -> {ok}"
    if t == "clarification":
        return f"ASK \"{e.get('question')}\" -> \"{e.get('response')}\""
    if t == "final_verification":
        return f"FINAL VERIFY passed={e.get('passed')}: {e.get('detail')}"
    if t == "search_found":
        return f"SEARCH found {e.get('target')}"
    if t == "perceive":
        amb = f", ambiguities: {e['ambiguities']}" if e.get("ambiguities") else ""
        return f"perceive frame {e.get('frame_id')}: {len(e.get('objects') or [])} objects{amb}"
    if t in ("limit", "safe_stop", "search_exhausted", "error"):
        return f"{t.upper()} " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("type", "sim_time"))
    return t


def wrap(draw, text, width, fnt):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def caption(img, header, lines, fnt=F_SMALL):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, H, W, H + CAPTION_H], fill=(20, 20, 24))
    y = H + 6
    for i, hl in enumerate(header):
        draw.text((8, y), hl, fill=(255, 255, 255) if i == 0 else (180, 180, 190), font=F_MED)
        y += 19
    for line in lines:
        for part in wrap(draw, line, W - 16, fnt):
            if y > H + CAPTION_H - 16:
                draw.text((8, y), "…", fill=(200, 200, 200), font=fnt)
                return
            color = (255, 110, 110) if "FAIL" in line or "ERROR" in line or "passed=False" in line else (220, 220, 220)
            draw.text((8, y), part, fill=color, font=fnt)
            y += 16


def card(lines, sub):
    img = Image.new("RGB", (W, H + CAPTION_H), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    y = 60
    for line in lines:
        for part in wrap(draw, line, W - 40, F_BIG):
            draw.text((20, y), part, fill=(255, 255, 255), font=F_BIG)
            y += 30
    y += 10
    for line in sub:
        for part in wrap(draw, line, W - 40, F_MED):
            color = (255, 110, 110) if "FAIL" in line or "ERROR" in line else (200, 200, 210)
            draw.text((20, y), part, fill=color, font=F_MED)
            y += 21
    return img


def detections(img, audits):
    draw = ImageDraw.Draw(img)
    if any(a.get("status") == "content_filtered" or "inappropriate content" in str(a.get("error", "")) for a in audits):
        draw.rectangle([0, 0, W, 30], fill=(160, 0, 0))
        draw.text((8, 6), "VLM PROVIDER CONTENT FILTER: frame refused (HTTP 400)", fill=(255, 255, 255), font=F_MED)
    scene = next((a["scene"] for a in reversed(audits) if a.get("scene") and not a.get("grounding")), None)
    scene = scene or next((a["scene"] for a in reversed(audits) if a.get("scene")), None)
    if not scene:
        return
    for g in scene.get("objects", []) + scene.get("regions", []):
        if not g.get("bbox_xyxy"):
            continue
        x1, y1, x2, y2 = g["bbox_xyxy"]
        col = STATUS_COLOR.get(g.get("status"), (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
        label = f"{g['instance_id']} {g.get('attributes', {}).get('color', '')} {g['name']} {g.get('status', '')[:5]}"
        tw = draw.textlength(label, font=F_SMALL)
        ty = max(0, y1 - 16)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + 15], fill=(0, 0, 0))
        draw.text((x1 + 3, ty), label, fill=col, font=F_SMALL)


def render(trial_dir: pathlib.Path, out: pathlib.Path, audit_dir: pathlib.Path) -> None:
    rec = json.loads((trial_dir / "trial_record.json").read_text())
    frames = {}  # frame_id -> (meta, rgb path, png sha as A computes it)
    for p in (trial_dir / "frames").glob("*_meta.json"):
        m = json.loads(p.read_text())
        f = trial_dir / "frames" / m["rgb_file"]
        frames[m["frame_id"]] = (m, f, a_png_sha(np.asarray(Image.open(f).convert("RGB"))))
    audits: dict[str, list[dict]] = {}  # by image hash: A may see one image under several frame_ids
    for a in episode_audits(audit_dir, {sha for _, _, sha in frames.values()}):
        audits.setdefault(a["image_sha256"], []).append(a)
        rgb = pathlib.Path(a.get("input_files", {}).get("rgb", ""))
        if a["frame_id"] not in frames and rgb.is_file() and a["image_sha256"] not in {x[2] for x in frames.values()}:
            frames[a["frame_id"]] = (json.loads(pathlib.Path(a["input_files"]["meta"]).read_text()), rgb,
                                     a["image_sha256"])
    metas = [frames[k][0] for k in sorted(frames)]
    events = rec["events"]
    ex = [e.get("result", e) for e in rec.get("extra", {}).get("execution_records", [])]
    fails = [x.get("error_code") for x in ex if not x.get("success")]
    slides = [(card([rec["trial_id"], f"\"{rec['instruction']}\""],
                    [f"perception: {rec['module_config']['perception'].split('.')[-1]}, "
                     f"planner: {rec['module_config']['planner'].split('.')[-1]}, "
                     f"executor: {rec['module_config']['executor'].split('.')[-1]}",
                     f"{len(metas)} head-camera frames; boxes = Student A's output for that frame "
                     "(green LOCALIZED, yellow UNLOCALIZED, magenta AMBIGUOUS)"]), CARD_S)]
    used = set()
    for m in metas:
        t = m["sim_time"]
        img = Image.new("RGB", (W, H + CAPTION_H))
        img.paste(Image.open(frames[m["frame_id"]][1]).convert("RGB").resize((W, H)), (0, 0))
        detections(img, audits.get(frames[m["frame_id"]][2], []))
        new = [i for i, e in enumerate(events) if i not in used and e.get("sim_time", 0) <= t + 1e-6
               and not (e["type"] == "perceive" and e.get("frame_id") != m["frame_id"])]
        used.update(new)
        shown = [event_line(events[i]) for i in new if events[i]["type"] != "perceive"]
        caption(img, [f"{rec['trial_id']}  frame {m['frame_id']}  sim t={t:.2f}s  camera={m['camera_name']}"],
                shown or ["(observation)"])
        slides.append((img, SLIDE_S if shown else QUIET_S))
    rest = [event_line(e) for i, e in enumerate(events)
            if (i not in used and e["type"] != "perceive") or e["type"] == "final_verification"]
    err = str(rec.get("error") or "").strip().splitlines()[-1:] or []
    slides.append((card([f"{rec['outcome']}",
                         f"claimed={rec['claimed_success']}  actual={rec['actual_success']}"],
                        rest[-8:] + ([f"ERROR: {err[0][:200]}"] if err else [])
                        + [f"failed action codes: {', '.join(map(str, fails)) or '-'}"]), CARD_S))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp = pathlib.Path(tmp)
        lines = []
        for i, (img, dur) in enumerate(slides):
            img.save(tmp / f"s{i:04d}.png")
            lines += [f"file 's{i:04d}.png'", f"duration {dur}"]
        lines.append(f"file 's{len(slides) - 1:04d}.png'")  # concat demuxer needs the last file twice
        (tmp / "list.txt").write_text("\n".join(lines) + "\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                        "-vf", "fps=10,format=yuv420p", "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                        str(out.resolve())], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial_dir", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--audit-dir", type=pathlib.Path, default=pathlib.Path("runs/night/audit_a"))
    args = ap.parse_args()
    render(args.trial_dir, args.out, args.audit_dir)
