#!/usr/bin/env python
"""Assemble the 3-5 minute submission demo video (0 API calls at build time).

Episodes 1-6 are complete, unedited Recorder videos of live end-to-end runs
(A: Student A perception on Qwen-VL, B: Qwen planner, C: Student C executor),
re-recorded on 2026-09-27 on the submission code (branch e2e), 47 live calls.
Episodes 7-8 show the bonus learned policies behind the same skill interface
(manipulation mode: GT perception + rule planner, 0 calls). The source clips
are in docs/submission/episodes/ (details: docs/submission/DEMO_NOTES.md).
Only caption cards are inserted BETWEEN episodes; nothing inside an episode is
cut, sped up or reordered.

    python scripts/make_demo_video.py            # -> docs/submission/EE4705_demo.mp4
"""

from __future__ import annotations

import pathlib
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
EP = ROOT / "docs/submission/episodes"
WORK = ROOT / "runs/demo_video"  # snap ffmpeg cannot write under /tmp
OUT = ROOT / "docs/submission/EE4705_demo.mp4"
W, H, FPS = 1440, 900, 10
BG, INK, MUTED, ACCENT, GOOD, WARN = (11, 18, 32), (236, 240, 246), (150, 164, 186), (232, 178, 92), (110, 214, 170), (240, 150, 120)
PURPLE = (170, 150, 240)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# (clip, card title, instruction, what to watch, colour of the tag, seconds on the card)
EPISODES = [
    ("1_f11_c_06_cube", "1 / 8  ·  Standard task", "Move the blue cube to the red area.",
     "A localizes the cube, B plans APPROACH › GRASP › MOVE_TO › PLACE › VERIFY › STOP, "
     "C executes each skill with closed-loop checks; the final verification confirms the placement.", GOOD, 4.5),
    ("2_f14_c_09_bottle", "2 / 8  ·  Different object", "Move the green bottle to the red area.",
     "Same pipeline on the tallest object, grasped and placed on the first attempt.", GOOD, 3.5),
    ("3_f23_c_2_08_cube", "3 / 8  ·  Scene variation", "Move the blue cube to the red area.",
     "Two stones are also on the table; the plan targets the cube instance that A grounded.", GOOD, 3.5),
    ("4_f04_smoke_4_search", "4 / 8  ·  Search (target not visible)", "move the stone onto the red region",
     "The robot starts facing away. B answers NEEDS_SEARCH, C turns and re-observes until A grounds the stone, "
     "then B re-plans and the task completes.", ACCENT, 4.5),
    ("5_f50_clarify_two_stones", "5 / 8  ·  Clarification", "put the stone on the red area",
     "Two stones match. B asks \"the gray one or the dark red one?\"; the user answers \"the gray one\" "
     "and the system moves that stone.", ACCENT, 4.5),
    ("6_f49_reject_stack", "6 / 8  ·  Infeasible request", "Stack the blue cube on top of the green bottle.",
     "B recognizes that stacking on objects is unsupported and refuses. No motion is executed.", ACCENT, 4.0),
    ("7_learned_c1_00", "7 / 8  ·  Bonus: learned grasp + learned place", "Move the blue cube to the red area.",
     "Both arm skills are driven by learned ACT policies behind the unchanged skill interface; the executor's "
     "post-condition checks are the same. (Bonus runs use ground-truth perception and the rule planner.)", PURPLE, 5.5),
    ("8_learned_c3_01_bottle", "8 / 8  ·  Bonus: learned grasp on the bottle", "Move the green bottle to the red area.",
     "The ACT grasp policy, trained with added bottle demonstrations, picks up the tall bottle.", PURPLE, 4.0),
]


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def card(path: pathlib.Path, title: str, lines: list[tuple[str, int, tuple, bool]], tag_color=ACCENT,
         footer="EE4705 Project 1.3  ·  simulated Unitree G1  ·  live end-to-end runs (A + B + C)") -> None:
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
    d.text((130, H - 70), footer, font=font(20), fill=MUTED)
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
        ("Six live end-to-end episodes recorded on the submission code, then two bonus episodes with "
         "learned policies. Each episode is one uncut recording; cards appear only between episodes.", 24, ACCENT, False),
        ("Left: observer camera (not an input).  Right: what A saw, and B's current plan.  Bottom: C's skill.",
         22, MUTED, False),
    ])
    still_clip(WORK / "c_intro.png", 8, WORK / "p00_intro.mp4")
    parts.append(WORK / "p00_intro.mp4")
    for k, (clip, title, instr, watch, color, secs) in enumerate(EPISODES, 1):
        png = WORK / f"c_{k}.png"
        foot = ("EE4705 Project 1.3  ·  bonus: learned ACT policies  ·  GT perception + rule planner"
                if color == PURPLE else "EE4705 Project 1.3  ·  simulated Unitree G1  ·  live end-to-end run (A + B + C)")
        card(png, title, [(f"“{instr}”", 32, INK, True), ("What to watch: " + watch, 26, MUTED, False)], color, foot)
        still_clip(png, secs, WORK / f"p{k:02d}a_card.mp4")
        reencode(EP / f"{clip}.mp4", WORK / f"p{k:02d}b_{clip}.mp4")
        parts += [WORK / f"p{k:02d}a_card.mp4", WORK / f"p{k:02d}b_{clip}.mp4"]
    card(WORK / "c_outro.png", "Summary", [
        ("Final evaluation, 50 live end-to-end trials (standard, scene and instruction variations, search, "
         "clarification, infeasible requests): 44 / 50 correct.", 28, INK, False),
        ("0 false success claims: when recovery runs out, the system reports failure instead of claiming success.",
         28, GOOD, False),
        ("Bonus: a learned ACT grasp scores 47 / 50 on the same 50 trials in manipulation mode "
         "(scripted grasp: 48 / 50), again with 0 false claims.", 26, PURPLE, False),
    ], GOOD, "EE4705 Project 1.3  ·  simulated Unitree G1")
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
