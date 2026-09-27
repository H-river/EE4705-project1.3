#!/usr/bin/env python
"""Assemble the 3-5 minute submission demo video (0 API calls).

Every episode clip is one complete, unedited Recorder video of a live end-to-end
run (A: Student A perception on Qwen-VL, B: Qwen planner, C: Student C executor)
from the final evaluation re-recordings on tag `final-40` (2026-09-24, see
docs/night_run/EPISODES.md). Only caption cards are inserted BETWEEN episodes;
nothing inside an episode is cut, sped up or reordered.

    python scripts/make_demo_video.py            # -> docs/demo/EE4705_demo.mp4
"""

from __future__ import annotations

import pathlib
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
EP = ROOT / "docs/night_run/episodes"
WORK = ROOT / "runs/demo_video"  # snap ffmpeg cannot write under /tmp
OUT = ROOT / "docs/demo/EE4705_demo.mp4"
W, H, FPS = 1440, 900, 10
BG, INK, MUTED, ACCENT, GOOD, WARN = (11, 18, 32), (236, 240, 246), (150, 164, 186), (232, 178, 92), (110, 214, 170), (240, 150, 120)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# (clip, card title, instruction, what to watch, colour of the tag, seconds on the card)
EPISODES = [
    ("final_ok_f11_c_06_cube", "1 / 7  ·  Standard task", "Move the blue cube to the red area.",
     "A localizes the cube, B plans APPROACH › GRASP › MOVE_TO › PLACE › VERIFY › STOP, "
     "C executes each skill with closed-loop checks; the final verification confirms the placement.", GOOD, 4.5),
    ("final_ok_f15_c_10_bottle", "2 / 7  ·  Different object", "Move the green bottle to the red area.",
     "Same pipeline on the tallest object; placement error 0.2 cm.", GOOD, 3.5),
    ("final_ok_f23_c_2_08_cube", "3 / 7  ·  Scene variation", "Move the blue cube to the red area.",
     "Two stones are also on the table; the plan still targets the cube instance A grounded.", GOOD, 3.5),
    ("final_ok_f04_smoke_4_search", "4 / 7  ·  Search (target not visible)", "move the stone onto the red region",
     "The robot starts facing away. B answers NEEDS_SEARCH, C turns and re-observes until A grounds the stone, "
     "then B re-plans and the task completes.", ACCENT, 4.5),
    ("final_ok_f50_clarify_two_stones", "5 / 7  ·  Clarification", "put the stone on the red area",
     "Two stones match. B asks \"the gray one or the dark red one?\"; the user answers \"the gray one\" "
     "and the system moves that stone.", ACCENT, 4.5),
    ("final_ok_f49_reject_stack", "6 / 7  ·  Infeasible request", "Stack the blue cube on top of the green bottle.",
     "B recognizes that stacking on objects is unsupported and refuses. No motion is executed.", ACCENT, 4.0),
    ("final_fail_f25_c_2_10_bottle", "7 / 7  ·  Retry and failure reporting", "Move the green bottle to the red area.",
     "After parking, the bottle leaves the head camera's view: GRASP reports TARGET_LOST, the system searches "
     "and retries, then stops at the plan limit and reports FAILURE instead of claiming success.", WARN, 5.0),
]


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def card(path: pathlib.Path, title: str, lines: list[tuple[str, int, tuple, bool]], tag_color=ACCENT) -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 250
    d.text((130, y), title, font=font(46, True), fill=INK)
    y += 90
    for text, size, color, bold in lines:
        for chunk in textwrap.wrap(text, width=int(1150 / (size * 0.52))) or [""]:
            d.text((130, y), chunk, font=font(size, bold), fill=color)
            y += int(size * 1.45)
        y += 14
    d.rectangle([90, 250, 98, max(650, y - 14)], fill=tag_color)
    d.text((130, H - 70), "EE4705 Project 1.3  ·  simulated Unitree G1  ·  live end-to-end runs (A + B + C)",
           font=font(20), fill=MUTED)
    im.save(path)


def still_clip(png: pathlib.Path, seconds: float, out: pathlib.Path) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", f"{seconds}",
                    "-i", str(png), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)], check=True)


def reencode(src: pathlib.Path, out: pathlib.Path) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", f"scale={W}:{H},fps={FPS}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)], check=True)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    card(WORK / "c_intro.png", "Language-guided tabletop manipulation", [
        ("Simulated Unitree G1 with a Robotiq gripper, following natural-language instructions.", 28, INK, False),
        ("A  perception: Qwen-VL grounding from the head camera", 26, MUTED, False),
        ("B  planning: Qwen LLM → structured skill plan (or search / clarify / refuse)", 26, MUTED, False),
        ("C  execution: closed-loop skills with post-condition checks", 26, MUTED, False),
        ("Seven complete, unedited episodes from the final evaluation run. Each episode is one uncut "
         "recording; cards appear only between episodes.", 24, ACCENT, False),
        ("Left: observer camera (not an input).  Right: what A saw, and B's current plan.  Bottom: C's skill.",
         22, MUTED, False),
    ])
    still_clip(WORK / "c_intro.png", 8, WORK / "p00_intro.mp4")
    parts.append(WORK / "p00_intro.mp4")
    for k, (clip, title, instr, watch, color, secs) in enumerate(EPISODES, 1):
        png = WORK / f"c_{k}.png"
        card(png, title, [(f"“{instr}”", 32, INK, True), ("What to watch: " + watch, 26, MUTED, False)], color)
        still_clip(png, secs, WORK / f"p{k:02d}a_card.mp4")
        reencode(EP / f"{clip}.mp4", WORK / f"p{k:02d}b_{clip}.mp4")
        parts += [WORK / f"p{k:02d}a_card.mp4", WORK / f"p{k:02d}b_{clip}.mp4"]
    card(WORK / "c_outro.png", "Summary", [
        ("Final evaluation: 50 trials (standard tasks, scene and instruction variations, search, clarification, "
         "infeasible requests). Best run: 44 / 50 correct.", 28, INK, False),
        ("0 false success claims: when a step fails and recovery runs out, the system reports failure.", 28, GOOD, False),
        ("Recovery shown: search when the target is not visible, clarification when ambiguous, "
         "retry after a lost target, refusal of infeasible requests.", 26, MUTED, False),
    ], GOOD)
    still_clip(WORK / "c_outro.png", 8, WORK / "p99_outro.mp4")
    parts.append(WORK / "p99_outro.mp4")
    lst = WORK / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", str(OUT)],
                   check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(OUT)],
                         capture_output=True, text=True).stdout.strip()
    print(f"{OUT}  {float(dur):.1f} s  {OUT.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
