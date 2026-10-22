# SPDX-License-Identifier: Apache-2.0
"""Scores each comparison run inside the LLM's decode window.

LLM: generated tokens / decode wall time. Training: A2C updates per second.
Game: rendered frames per second and the gaps between consecutive frames
(a smooth 30 Hz game has 33 ms gaps; anything over 50 ms is a visible stutter).
"""
import csv
import json
import os
import sys

RUNS = [
    ("cmp_llm_alone", "LLM alone (stock MPK, 128 workers)"),
    ("cmp_single128", "LLM alone, MPK single-kernel launch"),
    ("cmp_cotenants_alone", "Training + game alone (2 processes)"),
    ("cmp_procs", "Separate processes, time-sliced"),
    ("cmp_streams_128", "One process, separate kernels, stock MPK"),
    ("cmp_streams_96", "One process, separate kernels, MPK capped at 96"),
    ("cmp_mps", "Separate processes, MPS with SM split"),
    ("cmp_mega", "Megakernel (one launch)"),
    ("cmp_fused", "Co-tenants fused into MPK worker kernel"),
]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def llm_failed(d):
    log = f"{d}/log.txt"
    return os.path.exists(log) and "OverflowError" in open(log).read()


def score(d):
    run = json.load(open(f"{d}/run.json")) if os.path.exists(f"{d}/run.json") else None
    frames = read_csv(f"{d}/frames_0.csv")
    train = read_csv(f"{d}/train.csv")
    if run:
        t0, t1 = run["wall_start"] * 1e9, run["wall_end"] * 1e9
    else:
        src = frames or train
        t0, t1 = src[0]["wall_ns"] + 3e9, src[-1]["wall_ns"]
    out = {"window_s": (t1 - t0) / 1e9, "llm_failed": llm_failed(d)}
    if run:
        out["tok_s"] = run["generated"] / out["window_s"]
        out["ms_tok"] = 1000 / out["tok_s"]
    upd = [(r["wall_ns"], r["updates"]) for r in (train or frames)]
    inside = [u for u in upd if t0 <= u[0] <= t1]
    if len(inside) >= 2:
        out["upd_s"] = (inside[-1][1] - inside[0][1]) / ((inside[-1][0] - inside[0][0]) / 1e9)
    else:
        out["upd_s"] = 0.0
    ft = [r["wall_ns"] for r in frames if t0 <= r["wall_ns"] <= t1]
    out["fps"] = len(ft) / out["window_s"]
    # Include the stall from the window start to the first frame and from the
    # last frame to the window end: a game that never draws is frozen throughout.
    edges = [t0] + ft + [t1]
    gaps = [(b - a) / 1e6 for a, b in zip(edges, edges[1:])]
    out["gap_p50"] = pct(gaps, 0.5)
    out["gap_p99"] = pct(gaps, 0.99)
    out["gap_max"] = max(gaps)
    out["stutters"] = sum(g > 50 for g in gaps)
    return out


def main(root):
    rows = []
    for name, label in RUNS:
        d = os.path.join(root, name)
        if os.path.isdir(d):
            rows.append((name, label, score(d)))
    print(f"{'configuration':50s} {'ms/tok':>7s} {'tok/s':>6s} {'upd/s':>6s} {'fps':>5s} "
          f"{'gap p50':>8s} {'gap p99':>8s} {'gap max':>8s} {'>50ms':>6s}")
    for name, label, s in rows:
        if s["llm_failed"]:
            label += " [LLM OUTPUT CORRUPTED]"
        print(f"{label:50s} {s.get('ms_tok', float('nan')):7.2f} {s.get('tok_s', float('nan')):6.1f} "
              f"{s['upd_s']:6.0f} {s['fps']:5.1f} {s['gap_p50']:8.1f} {s['gap_p99']:8.1f} "
              f"{s['gap_max']:8.0f} {s['stutters']:6d}")
    json.dump({n: s for n, _, s in rows}, open(os.path.join(root, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUNS", "/root/runs"))
