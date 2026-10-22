# SPDX-License-Identifier: Apache-2.0
"""Composes a video of one megakernel launch.

Left: the game exactly as the kernel rendered it. Right: the block layout of the
launch, live rates for all three workloads, and the LLM's text as it was being
decoded at that frame.

    python bench/compose.py <run_dir>

A run_dir holds run.json, frames_0.csv (or video_0.csv) and game_0.mp4 from
COTENANT_OUT; the text comes from the per-frame decode step the kernel logged.
"""
import argparse
import csv
import json
import os
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
GW, GH = 288, 512
SCALE = H / GH
LEFT = int(GW * SCALE)
X0 = LEFT + 28
STEPS_PER_LEARNER_UPDATE = 256 * 16  # environments per learner x rollout length
FONT_DIR = "/usr/share/fonts/truetype/dejavu"  # DejaVu, overridable with --fonts
FONT = "DejaVuSans.ttf"
BOLD = "DejaVuSans-Bold.ttf"
MONO = "DejaVuSansMono.ttf"
WRAP = 92  # characters per line of LLM output

BG = (16, 18, 24)
FG = (232, 234, 240)
DIM = (140, 146, 160)
ROLE = {
    "llm": (86, 156, 255),
    "sched": (120, 124, 136),
    "train": (255, 150, 60),
    "game": (120, 210, 90),
}


def load_fonts(directory):
    def font(name, size):
        return ImageFont.truetype(os.path.join(directory, name), size)

    return {"title": font(BOLD, 30), "sub": font(FONT, 17), "head": font(BOLD, 18),
            "stat": font(BOLD, 26), "lab": font(FONT, 14), "mono": font(MONO, 15)}


def wrap(text):
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, WRAP) or [""])
    return lines


def draw(game, a, f, cols, header, lines):
    """One output frame: `game` (GW x GH) on the left; on the right the block
    map, `cols` of (heading, colour, [(value, label)]), and the LLM's text
    under `header`."""
    img = Image.new("RGB", (W, H), BG)
    img.paste(game.resize((LEFT, H), Image.NEAREST), (0, 0))
    d = ImageDraw.Draw(img)
    d.text((X0, 18), "One CUDA kernel. One H100.", font=f["title"], fill=FG)
    d.text((X0, 58), "Mirage MPK megakernel running LLM inference, RL training and a game at the same time",
           font=f["sub"], fill=DIM)

    # Block map: one cell per thread block of the single launch.
    roles = (["llm"] * a.workers + ["sched"] * a.sched_blocks +
             ["game"] + ["train"] * (a.cotenant - 1))
    y = 96
    cell, gap, per_row = 10, 2, 66
    for b, role in enumerate(roles):
        cx = X0 + (b % per_row) * (cell + gap)
        cy = y + (b // per_row) * (cell + gap)
        d.rectangle([cx, cy, cx + cell, cy + cell], fill=ROLE[role])
    y += 2 * (cell + gap) + 8
    legend = [("llm", f"{a.workers} LLM decode workers"), ("sched", f"{a.sched_blocks} schedulers"),
              ("train", f"{a.cotenant - 1} RL trainers"), ("game", "1 game + renderer")]
    lx = X0
    for role, label in legend:
        d.rectangle([lx, y + 3, lx + 10, y + 13], fill=ROLE[role])
        d.text((lx + 16, y), label, font=f["lab"], fill=FG)
        lx += 26 + int(d.textlength(label, font=f["lab"])) + 14

    # Live stats for each workload.
    y += 34
    cw = (W - X0 - 20) // 3
    for c, (head, color, stats) in enumerate(cols):
        cx = X0 + c * cw
        d.rectangle([cx, y, cx + 4, y + 172], fill=color)
        d.text((cx + 14, y), head, font=f["head"], fill=color)
        for k, (val, lab) in enumerate(stats):
            sy = y + 30 + k * 48
            d.text((cx + 14, sy), val, font=f["stat"], fill=FG)
            d.text((cx + 14, sy + 30), lab, font=f["lab"], fill=DIM)
    y += 190

    # The LLM's output so far, its newest lines.
    d.text((X0, y), header, font=f["head"], fill=ROLE["llm"])
    y += 30
    max_lines = (H - y - 14) // 19
    for k, line in enumerate(lines[-max_lines:]):
        d.text((X0, y + k * 19), line, font=f["mono"], fill=FG)
    return img


def encode(out):
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                            stdin=subprocess.PIPE)


def rate(rows, i, key, window_ns=1_000_000_000):
    j = i
    while j > 0 and rows[i]["host_ns"] - rows[j - 1]["host_ns"] <= window_ns:
        j -= 1
    dt = (rows[i]["host_ns"] - rows[j]["host_ns"]) / 1e9
    return (rows[i][key] - rows[j][key]) / dt if dt > 0 else 0.0


def compose_run(a, f):
    from transformers import AutoTokenizer

    run = json.load(open(f"{a.source}/run.json"))
    tok = AutoTokenizer.from_pretrained(a.model)
    plen, ids = run["prompt_length"], run["token_ids"]
    log = f"{a.source}/video_0.csv" if os.path.exists(f"{a.source}/video_0.csv") else f"{a.source}/frames_0.csv"
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(log))]
    steps_per_update = (a.cotenant - 1) * STEPS_PER_LEARNER_UPDATE

    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", f"{a.source}/game_0.mp4",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out = a.out or f"{a.source}/showcase.mp4"
    enc = encode(out)
    text_cache = (-1, [])
    written = 0
    for i, r in enumerate(rows):
        raw = dec.stdout.read(GW * GH * 3)
        if len(raw) < GW * GH * 3:
            break
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
        step = int(r["decode_step"])
        if step != text_cache[0]:
            text = tok.decode(ids[plen:min(len(ids), step + 1)], skip_special_tokens=True)
            text_cache = (step, wrap(text))
        header = f"LLM output (decoded inside the same kernel)   t = {r['host_ns'] / 1e9:5.1f} s"
        img = draw(Image.frombytes("RGB", (GW, GH), raw), a, f, cols, header, text_cache[1])
        enc.stdin.write(img.tobytes())
        written += 1
    enc.stdin.close()
    enc.wait()
    dec.kill()
    print(f"wrote {out} ({written} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="the launch's run_dir")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--sched-blocks", type=int, default=4)
    ap.add_argument("--cotenant", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fonts", default=FONT_DIR, help="directory with the DejaVu TTF fonts")
    a = ap.parse_args()
    f = load_fonts(a.fonts)
    compose_run(a, f)


if __name__ == "__main__":
    main()
