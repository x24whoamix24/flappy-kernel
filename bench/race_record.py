# SPDX-License-Identifier: Apache-2.0
"""Records a race against the AI for docs/media/race.gif, next to the server
(the WebSocket then adds no lag):

    python bench/race_record.py ws://localhost:8080/ws <out dir> <seconds>

It joins a race, asks the LLM a question a second in, and flies the bird for
<seconds>: it finds its bird, the AI's and the next pipe's gap in each frame
and flaps to stay close to the AI's bird without leaving the gap. Then it
stops flapping, so the recording ends with the crash. Every frame goes to
<out dir>/f00000.jpg..., and the server's numbers at that frame to
<out dir>/states.jsonl, with the trainer's pipes per practice game read
from the shared block (/dev/shm/flappy, at server/layout.json's offset);
bench/compose.py --race <out dir> lays them out."""
import asyncio
import io
import json
import mmap
import struct
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image

QUESTION = ("Write a short story, about 400 words, about a little bird that learns to fly "
            "through pipes inside a GPU, while a neural network beside it learns faster.")
BIRD_X0, BIRD_X1, GROUND_Y, BIRD_H = 57, 91, 404, 24
FOLLOW = 34  # how far below the AI's bird to flap


def bird_top(rgb, test):
    """Top of the bird whose pixels pass `test`, or None (sprites have a 2 px
    outline above their colour)."""
    reg = rgb[:GROUND_Y, BIRD_X0 - 6:BIRD_X1 + 6].astype(int)
    ys = np.nonzero(test(reg[..., 0], reg[..., 1], reg[..., 2]))[0]
    return float(ys.min()) - 2 if len(ys) > 10 else None


def player_top(rgb):
    return bird_top(rgb, lambda r, g, b: (r > 200) & (g > 150) & (b < 120))


def ai_top(rgb):
    return bird_top(rgb, lambda r, g, b: (b > 235) & (r < 170) & (g > 150))


def gap_top(rgb):
    """Gap top of the first pipe that reaches the bird or lies ahead of it,
    measured at the pipe's right edge, clear of the birds."""
    img = rgb.astype(int)
    green = (img[..., 1] > 120) & (img[..., 1] > img[..., 0] + 30) & (img[..., 2] < 120)
    sky = (img[..., 2] > 170) & (img[..., 1] > 170) & (img[..., 0] < 150)
    is_pipe = green[:76].sum(axis=0) > 30
    for x in range(BIRD_X0, rgb.shape[1] - 3):
        if is_pipe[x]:
            right = x
            while right + 1 < len(is_pipe) and is_pipe[right + 1]:
                right += 1
            col = sky[:GROUND_Y, max(right - 3, x)]
            for y in range(76, 240):
                if col[y:y + 6].all():
                    return y
            return None
    return None


def practice_pipes():
    """Reads the trainer's mean score per practice game from the shared block."""
    layout = json.load(open(Path(__file__).resolve().parent.parent / "server" / "layout.json"))
    shm = open("/dev/shm/flappy", "rb")
    block = mmap.mmap(shm.fileno(), 0, access=mmap.ACCESS_READ)
    return lambda: struct.unpack_from("<f", block, layout["train_mean_score"])[0]


async def main(url, out, seconds):
    train_mean = practice_pipes()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    states = open(out / "states.jsonl", "w")
    async with aiohttp.ClientSession() as s, s.ws_connect(url) as ws:
        await ws.send_json({"t": "join", "mode": "race"})
        n, phase, me, state = 0, 0, {}, {}
        start = asked = stop = None
        last, prev, gap = 0.0, None, None
        async for msg in ws:
            now = time.time()
            if msg.type == aiohttp.WSMsgType.TEXT:
                d = json.loads(msg.data)
                if d.get("t") == "state":
                    # The finished answer is only in the history: the current
                    # one is no longer reported once its last tokens arrive.
                    answer = next((h for h in reversed(d.get("history") or [])
                                   if h.get("prompt") == QUESTION), None)
                    state = {"stats": d["stats"], "llm": d["llm"], "answer": answer}
                elif d.get("t") == "me" and d.get("me"):
                    me = d["me"]
                    phase = me["phase"]
                    if phase == 1 and now - last > 1.0:     # ready: start the race
                        await ws.send_json({"t": "flap"})
                        last = now
                    if phase == 2 and start is None:
                        start = now
                    if phase == 3 and start is not None and stop is None:
                        stop = now
                continue
            if start is None:
                continue
            (out / f"f{n:05d}.jpg").write_bytes(msg.data)
            states.write(json.dumps({"t": now - start, "me": me, "train_mean": train_mean(),
                                     **state}) + "\n")
            n += 1
            if stop is not None and now - stop > 2.5:
                break
            if asked is None and now - start > 1.0:
                await ws.send_json({"t": "prompt", "text": QUESTION})
                asked = now
            if phase != 2 or now - start > seconds:
                continue
            rgb = np.asarray(Image.open(io.BytesIO(msg.data)).convert("RGB"))
            top, ai, found = player_top(rgb), ai_top(rgb), gap_top(rgb)
            gap = found if found is not None else gap
            vy = top - prev if top is not None and prev is not None else 0.0
            # A flap lifts the bird 45 px: flap between gap+50 (its peak stays
            # below the gap's top) and gap+62 (before it reaches the bottom;
            # gap 100 px, bird 24 px), as close to the AI's bird as that allows.
            limit = gap + 100 - BIRD_H - 14 if gap is not None else 220
            if ai is not None:
                limit = min(limit, ai + FOLLOW)
            if gap is not None:
                limit = max(limit, gap + 50)
            if top is not None and top > limit and vy >= 0 and now - last > 0.1:
                await ws.send_json({"t": "flap"})
                last = now
            prev = top
    print(f"frames {n}, result: you {me.get('you_score')} AI {me.get('ai_score')}")


asyncio.run(main(sys.argv[1], sys.argv[2], float(sys.argv[3])))
