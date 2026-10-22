# SPDX-License-Identifier: Apache-2.0
"""Composes the showcase video from one megakernel launch.

Left: the game exactly as the kernel rendered it. Right: the block layout of the
launch, live rates for all three workloads, and the LLM's text as it was being
decoded at that frame (from the per-frame decode step the kernel logged).
"""
import argparse
import csv
import json
import os
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

W, H = 1280, 720
GW, GH = 288, 512
SCALE = H / GH
LEFT = int(GW * SCALE)
STEPS_PER_LEARNER_UPDATE = 256 * 16  # environments per learner x rollout length
FONT_DIR = "/usr/share/fonts/truetype/dejavu"  # DejaVu, overridable with --fonts
FONT = "DejaVuSans.ttf"
BOLD = "DejaVuSans-Bold.ttf"
MONO = "DejaVuSansMono.ttf"

BG = (16, 18, 24)
FG = (232, 234, 240)
DIM = (140, 146, 160)
ROLE = {
    "llm": (86, 156, 255),
    "sched": (120, 124, 136),
    "train": (255, 150, 60),
    "game": (120, 210, 90),
}


def rate(rows, i, key, window_ns=1_000_000_000):
    j = i
    while j > 0 and rows[i]["host_ns"] - rows[j - 1]["host_ns"] <= window_ns:
        j -= 1
    dt = (rows[i]["host_ns"] - rows[j]["host_ns"]) / 1e9
    return (rows[i][key] - rows[j][key]) / dt if dt > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--sched-blocks", type=int, default=4)
    ap.add_argument("--cotenant", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fonts", default=FONT_DIR, help="directory with the DejaVu TTF fonts")
    a = ap.parse_args()

    run = json.load(open(f"{a.run_dir}/run.json"))
    tok = AutoTokenizer.from_pretrained(a.model)
    plen = run["prompt_length"]
    ids = run["token_ids"]
    rows = []
    log = f"{a.run_dir}/video_0.csv" if os.path.exists(f"{a.run_dir}/video_0.csv") else f"{a.run_dir}/frames_0.csv"
    for r in csv.DictReader(open(log)):
        rows.append({k: float(v) for k, v in r.items()})

    f_title = ImageFont.truetype(os.path.join(a.fonts, BOLD), 30)
    f_sub = ImageFont.truetype(os.path.join(a.fonts, FONT), 17)
    f_head = ImageFont.truetype(os.path.join(a.fonts, BOLD), 18)
    f_stat = ImageFont.truetype(os.path.join(a.fonts, BOLD), 26)
    f_lab = ImageFont.truetype(os.path.join(a.fonts, FONT), 14)
    f_mono = ImageFont.truetype(os.path.join(a.fonts, MONO), 15)

    steps_per_update = (a.cotenant - 1) * STEPS_PER_LEARNER_UPDATE
    roles = (["llm"] * a.workers + ["sched"] * a.sched_blocks +
             ["game"] + ["train"] * (a.cotenant - 1))

    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", f"{a.run_dir}/game_0.mp4",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out = a.out or f"{a.run_dir}/showcase.mp4"
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                           stdin=subprocess.PIPE)
    x0 = LEFT + 28
    text_cache = (-1, [])
    written = 0
    for i, r in enumerate(rows):
        raw = dec.stdout.read(GW * GH * 3)
        if len(raw) < GW * GH * 3:
            break
        img = Image.new("RGB", (W, H), BG)
        game = Image.frombytes("RGB", (GW, GH), raw).resize((LEFT, H), Image.NEAREST)
        img.paste(game, (0, 0))
        d = ImageDraw.Draw(img)
        d.text((x0, 18), "One CUDA kernel. One H100.", font=f_title, fill=FG)
        d.text((x0, 58), "Mirage MPK megakernel running LLM inference, RL training and a game at the same time",
               font=f_sub, fill=DIM)

        # Block map: one cell per thread block of the single launch.
        y = 96
        cell, gap, per_row = 12, 2, 66
        for b, role in enumerate(roles):
            cx = x0 + (b % per_row) * (cell + gap)
            cy = y + (b // per_row) * (cell + gap)
            d.rectangle([cx, cy, cx + cell, cy + cell], fill=ROLE[role])
        y += 2 * (cell + gap) + 8
        legend = [("llm", f"{a.workers} LLM decode workers"), ("sched", f"{a.sched_blocks} schedulers"),
                  ("train", f"{a.cotenant - 1} RL trainers"), ("game", "1 game + renderer")]
        lx = x0
        for role, label in legend:
            d.rectangle([lx, y + 3, lx + 10, y + 13], fill=ROLE[role])
            d.text((lx + 16, y), label, font=f_lab, fill=FG)
            lx += 26 + int(d.textlength(label, font=f_lab)) + 14

        # Live stats for each workload.
        y += 34
        t = r["host_ns"] / 1e9
        cols = [
            ("INFERENCE  Qwen3-8B", ROLE["llm"], [
                (f"{rate(rows, i, 'decode_step'):.0f}", "tokens / s"),
                (f"{max(0, int(r['decode_step']) - plen + 1)}", "tokens generated"),
            ]),
            ("TRAINING  A2C policy", ROLE["train"], [
                (f"{rate(rows, i, 'updates'):.0f}", "training steps / s"),
                (f"{rate(rows, i, 'updates') * steps_per_update / 1e6:.0f}M", "game steps simulated / s"),
                (f"{r['train_mean']:.0f}", "pipes per practice game"),
            ]),
            ("GAME  Flappy Kernel", ROLE["game"], [
                (f"{rate(rows, i, 'frame'):.0f}", "frames / s"),
                (f"{int(r['game_score'])}", "score"),
                (f"{int(r['game_best'])}", "best"),
            ]),
        ]
        cw = (W - x0 - 20) // 3
        for c, (head, color, stats) in enumerate(cols):
            cx = x0 + c * cw
            d.rectangle([cx, y, cx + 4, y + 172], fill=color)
            d.text((cx + 14, y), head, font=f_head, fill=color)
            for k, (val, lab) in enumerate(stats):
                sy = y + 30 + k * 48
                d.text((cx + 14, sy), val, font=f_stat, fill=FG)
                d.text((cx + 14, sy + 30), lab, font=f_lab, fill=DIM)
        y += 190

        # LLM output as of this frame.
        d.text((x0, y), f"LLM output (decoded inside the same kernel)   t = {t:5.1f} s", font=f_head, fill=ROLE["llm"])
        y += 30
        step = int(r["decode_step"])
        if step != text_cache[0]:
            text = tok.decode(ids[plen:min(len(ids), step + 1)], skip_special_tokens=True)
            lines = []
            for para in text.split("\n"):
                lines.extend(textwrap.wrap(para, 92) or [""])
            text_cache = (step, lines)
        lines = text_cache[1]
        max_lines = (H - y - 14) // 19
        for k, line in enumerate(lines[-max_lines:]):
            d.text((x0, y + k * 19), line, font=f_mono, fill=FG)
        enc.stdin.write(img.tobytes())
        written += 1
    enc.stdin.close()
    enc.wait()
    dec.kill()
    print(f"wrote {out} ({written} frames)")


if __name__ == "__main__":
    main()
