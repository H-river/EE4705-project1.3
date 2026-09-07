"""Observer-only recording. No rendering or truth is fed back into A/B/C."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

from core.rendering import mujoco
from core.world import SimWorld
from eval.logger import to_json_safe


class RecordedWorld(SimWorld):
    def __init__(self):
        self.recorder = None
        self._record_steps = 0
        super().__init__(grasp_mode="weld")

    def step(self, n=1):
        if self.recorder is None:
            return super().step(n)
        interval = max(1, round(1 / (self.recorder.fps * self.timestep)))
        while n > 0:
            count = min(n, interval - self._record_steps)
            super().step(count)
            n -= count
            self._record_steps += count
            if self._record_steps == interval:
                self._record_steps = 0
                self.recorder.capture()


class Recorder:
    WIDTH, HEIGHT = 1440, 900

    def __init__(self, world, out, instruction, scenario, fps=10, video=True):
        self.world, self.out, self.fps = world, pathlib.Path(out), fps
        self.instruction, self.scenario = instruction, scenario
        self.events, self.frames, self.count = [], [], 0
        self.scene = self.plan = self.obs = None
        self.stage, self.action, self.message = "READY", None, "Starting the episode"
        self.actual = None
        self.module_modes = "A: demo | B: demo | C: demo"
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [.18, 0, .95]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.25, 130, -22
        self.font = self._font(19)
        self.small = self._font(16)
        self.large = self._font(30)
        self.encoder = None
        self._encoder_log = None
        if video:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg is required for video; install it or use --no-video")
            self._encoder_log = (self.out / "ffmpeg.log").open("w")
            self.encoder = subprocess.Popen([
                ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", f"{self.WIDTH}x{self.HEIGHT}",
                "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.out / "episode.mp4"),
            ], stdin=subprocess.PIPE, stderr=self._encoder_log)

    @staticmethod
    def _font(size):
        path = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)

    def event(self, kind, data, hold=0):
        entry = {"type": kind, "sim_time": self.world.sim_time,
                 "video_time": self.count / self.fps, "data": to_json_safe(data)}
        self.events.append(entry)
        if hold:
            self.capture(hold=hold)

    def on_scene(self, obs, scene):
        self.obs, self.scene = obs, scene
        previous = self.stage
        self.stage = "A / PERCEIVE"
        self.event("A.scene", scene, hold=.6 if self.plan is None else .2)
        self.stage = previous

    def on_plan(self, instruction, scene, plan):
        self.plan, self.stage = plan, "B / PLAN"
        self.message = plan.reason or "Validated action order is checked by the backbone next"
        self.event("B.plan", {"instruction": instruction, "scene_frame_id": scene.frame_id, "plan": plan}, hold=1.5)

    def on_action(self, phase, action, result):
        self.action, self.stage = action, "C / " + action.skill.value
        self.message = ("Executing with public robot controls" if result is None else
                        ("OK" if result.success else result.error_code.value) + ": " + str(result.info.get("detail", "")))
        self.event("C." + phase, {"action": action, "result": result},
                   hold=.4 if result is None else (1.2 if not result.success else .3))

    def _compose(self):
        canvas = Image.new("RGB", (self.WIDTH, self.HEIGHT), "#0d1523")
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 18), "ABC / TABLETOP WORKFLOW", font=self.large, fill="#eff5ff")
        draw.text((28, 60), self.instruction[:100], font=self.font, fill="#c0cfdf")
        draw.text((28, 93), "WORKFLOW DEMO   |   " + self.module_modes + "   |   simulated robot / weld attachment", font=self.small, fill="#e8ba71")
        with self.world.lock:
            self.world._refresh_kinematics()
            renderer = self.world._get_renderer()
            renderer.disable_depth_rendering()
            renderer.update_scene(self.world.data, camera=self.camera)
            observer = Image.fromarray(renderer.render().copy())
        canvas.paste(observer.resize((784, 588)), (24, 150))
        draw.text((30, 125), "LIVE SIMULATION / observer camera (not an A input)", font=self.small, fill="#a3b6d0")
        draw.rounded_rectangle((830, 148, 1416, 490), 12, fill="#172337")
        draw.text((848, 159), "A / last analyzed head-camera image", font=self.font, fill="#70d4cf")
        if self.obs is not None:
            camera = Image.fromarray(self.obs.rgb)
            boxdraw = ImageDraw.Draw(camera)
            for obj in self.scene.objects + self.scene.regions:
                if obj.bbox_xyxy:
                    boxdraw.rectangle(obj.bbox_xyxy, outline="#67fff0", width=2)
                    x, y = obj.bbox_xyxy[:2]
                    boxdraw.text((x + 3, max(0, y - 19)), f"{obj.instance_id} {obj.name}", font=self.small, fill="#67fff0", stroke_width=1, stroke_fill="black")
            canvas.paste(camera.resize((384, 288)), (842, 195))
            draw.text((1235, 203), f"frame {self.obs.frame_id}", font=self.small, fill="white")
            draw.text((1235, 229), f"t={self.obs.sim_time:.2f}s", font=self.small, fill="#b6c5d8")
            draw.text((1235, 265), "Held until", font=self.small, fill="#b6c5d8")
            draw.text((1235, 288), "A runs again", font=self.small, fill="#b6c5d8")
        draw.rounded_rectangle((830, 505, 1416, 735), 12, fill="#172337")
        draw.text((848, 517), "B / structured plan", font=self.font, fill="#97b6ff")
        if self.plan:
            for i, action in enumerate(self.plan.actions[:7]):
                active = self.action == action
                label = f"{i+1:02d}  {action.skill.value:<10}  {action.target or ''}"
                draw.text((854, 553 + i * 25), label, font=self.small, fill="#ffe19f" if active else "#b6c5d8")
        draw.rounded_rectangle((24, 751, 1416, 876), 12, fill="#172337")
        draw.text((42, 766), self.stage, font=self.large, fill="#78ddbd")
        draw.text((750, 774), f"sim {self.world.sim_time:.2f}s   |   attachment: {'YES' if self.world.is_attached() else 'NO'}", font=self.font, fill="white")
        for i, line in enumerate(textwrap.wrap(self.message, width=125)[:2]):
            draw.text((42, 811+i*23), line, font=self.small, fill="#d4dfef")
        return canvas

    def capture(self, hold=0, snapshot=None):
        # A display hold repeats one physical state. It does not step physics.
        # Both clocks and every hold are explicit in episode.json.
        frame = self._compose()
        repeat = max(1, round(hold*self.fps))
        if snapshot:
            frame.save(self.out / snapshot)
        if self.encoder:
            raw = frame.tobytes()
            for _ in range(repeat):
                self.encoder.stdin.write(raw)
        self.frames.append({"frame": self.count, "count": repeat,
                            "video_time": self.count / self.fps,
                            "sim_time": self.world.sim_time, "display_hold": bool(hold)})
        self.count += repeat

    def close(self):
        if self.encoder:
            self.encoder.stdin.close()
            code = self.encoder.wait(timeout=60)
            self.encoder = None
            self._encoder_log.close()
            if code:
                raise RuntimeError(f"ffmpeg failed ({code}); see {self.out / 'ffmpeg.log'}")

    def save(self, summary):
        summary = to_json_safe({**summary, "events": self.events, "frames": self.frames,
                               "fps": self.fps, "video_duration_s": self.count / self.fps,
                               "video_available": (self.out / "episode.mp4").exists(),
                               "timing_note": "Video includes labeled display holds. sim_time is physics time; video_time includes holds. This is not an uncut assessment recording."})
        payload = json.dumps(summary, indent=2, allow_nan=False)
        (self.out / "episode.json").write_text(payload)
        template = pathlib.Path(__file__).with_name("replay.html").read_text()
        (self.out / "index.html").write_text(template.replace("__EPISODE_JSON__", payload.replace("<", "\\u003c")))
        return summary
