# SPDX-License-Identifier: Apache-2.0
"""Side-by-side video of the same three workloads under different co-location
setups. Each panel shows the game exactly as recorded (wall-clock paced, so a
stalled game is a frozen picture), aligned to the start of that run's LLM
decode, with live training and game rates and the run's measured LLM speed.
"""
import argparse
import bisect
import csv
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
GW, GH = 288, 512
FONT_DIR = "/usr/share/fonts/truetype/dejavu"  # DejaVu, overridable with --fonts
FONT = "DejaVuSans.ttf"
BOLD = "DejaVuSans-Bold.ttf"
BG, FG, DIM = (16, 18, 24), (232, 234, 240), (140, 146, 160)
GOOD, BAD = (120, 210, 90), (255, 107, 90)

PANELS = [
    ("cmp_streams_96", "Separate kernels", "one process, two streams"),
    ("cmp_procs", "Separate processes", "default GPU time-slicing"),
    ("cmp_mps", "Separate processes + MPS", "SM split tuned by hand"),
    ("cmp_fused", "Co-located in Mirage", "one MPK launch"),
]


def frames(path):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out = []
    while True:
        b = p.stdout.read(GW * GH * 3)
        if len(b) < GW * GH * 3:
            break
        out.append(b)
    return out


def load(run_dir):
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(f"{run_dir}/video_0.csv"))]
    imgs = frames(f"{run_dir}/game_0.mp4")
    n = min(len(rows), len(imgs))
    rows, imgs = rows[:n], imgs[:n]
    run = json.load(open(f"{run_dir}/run.json")) if os.path.exists(f"{run_dir}/run.json") else None
    if run:
        t0 = run["wall_start"] * 1e9
    else:
        t0 = rows[int(n * 0.3)]["wall_ns"]
    train = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(f"{run_dir}/train.csv"))] \
        if os.path.exists(f"{run_dir}/train.csv") else None
    failed = os.path.exists(f"{run_dir}/log.txt") and "OverflowError" in open(f"{run_dir}/log.txt").read()
    return {"rows": rows, "imgs": imgs, "t": [r["wall_ns"] for r in rows], "t0": t0, "run": run,
            "train": train, "failed": failed}


def rate_at(series, t, key):
    ts = [r["wall_ns"] for r in series]
    i = bisect.bisect_right(ts, t) - 1
    j = bisect.bisect_right(ts, t - 1e9) - 1
    if i <= 0 or j < 0 or i == j:
        return 0.0
    return (series[i][key] - series[j][key]) / ((series[i]["wall_ns"] - series[j]["wall_ns"]) / 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fonts", default=FONT_DIR, help="directory with the DejaVu TTF fonts")
    a = ap.parse_args()
    comp = json.load(open(f"{a.root}/compare.json"))
    runs = [(load(f"{a.root}/{name}"), name, title, sub) for name, title, sub in PANELS]

    f_title = ImageFont.truetype(os.path.join(a.fonts, BOLD), 34)
    f_sub = ImageFont.truetype(os.path.join(a.fonts, FONT), 18)
    f_head = ImageFont.truetype(os.path.join(a.fonts, BOLD), 24)
    f_lab = ImageFont.truetype(os.path.join(a.fonts, FONT), 16)
    f_val = ImageFont.truetype(os.path.join(a.fonts, BOLD), 22)
    pw = W // len(runs)
    scale = 1.5
    gw, gh = int(GW * scale), int(GH * scale)
    out = a.out or f"{a.root}/comparison.mp4"
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                           stdin=subprocess.PIPE)
    for k in range(int(a.seconds * FPS)):
        tau = k / FPS
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((24, 14), "One H100: Qwen3-8B request + RL training + live game", font=f_title, fill=FG)
        d.text((24, 56), f"Same three workloads, four ways to run them   t = {tau:4.1f} s into the LLM request",
               font=f_sub, fill=DIM)
        for p, (r, name, title, sub) in enumerate(runs):
            x0 = p * pw
            t = r["t0"] + tau * 1e9
            i = max(0, bisect.bisect_right(r["t"], t) - 1)
            game = Image.frombytes("RGB", (GW, GH), r["imgs"][i]).resize((gw, gh), Image.NEAREST)
            img.paste(game, (x0 + (pw - gw) // 2, 156))
            if r["rows"][i]["frame"] == 0:
                msg = "no frame yet: no free SMs"
                d.text((x0 + (pw - d.textlength(msg, font=f_head)) // 2, 156 + gh // 2), msg,
                       font=f_head, fill=BAD)
            d.text((x0 + 24, 90), title, font=f_head, fill=FG)
            d.text((x0 + 24, 124), sub, font=f_lab, fill=DIM)
            s = comp.get(name, {})
            series = r["train"] or r["rows"]
            upd = rate_at(series, t, "updates")
            seqs = r["rows"]
            fps = rate_at(seqs, t, "frame")
            y = 156 + gh + 14
            if r["failed"]:
                llm, llm_ok = "corrupt", False
            else:
                llm, llm_ok = f"{s.get('tok_s', 0):.0f} tok/s", True
            stats = [("LLM", llm, llm_ok), ("Training", f"{upd:.0f} upd/s", upd > 50),
                     ("Game", f"{fps:.0f} fps", fps > 25)]
            for q, (lab, val, ok) in enumerate(stats):
                sx = x0 + 24 + q * ((pw - 48) // 3)
                d.text((sx, y), lab, font=f_lab, fill=DIM)
                d.text((sx, y + 20), val, font=f_val, fill=GOOD if ok else BAD)
            if p:
                d.line([(x0, 90), (x0, H - 10)], fill=(40, 44, 56), width=2)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
