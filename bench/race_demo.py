# SPDX-License-Identifier: Apache-2.0
"""Races the AI in one arcade slot and saves every frame it receives, for
docs/media/race.gif. Run it next to the server (on the same host), where the
WebSocket adds no lag:

    python bench/race_demo.py ws://localhost:8080/ws <out dir> <seconds>

It follows the server's phase reports (1 ready, 2 playing, 3 over). While
playing it finds its bird and the next pipe's gap in each frame and flaps
when the bird falls toward the bottom of the gap. out/log.csv lists each
race's result; ffmpeg turns the frames of one race into the GIF."""
import asyncio, io, json, sys, time
from pathlib import Path
import aiohttp
import numpy as np
from PIL import Image

BIRD_X0, BIRD_X1, GROUND_Y, BIRD_H = 57, 91, 404, 24

def bird_top(rgb):
    """The top of the yellow bird's hitbox (its sprite has a 2 px outline)."""
    reg = rgb[:GROUND_Y, BIRD_X0 - 6:BIRD_X1 + 6].astype(int)
    yellow = (reg[..., 0] > 200) & (reg[..., 1] > 150) & (reg[..., 2] < 120)
    ys = np.nonzero(yellow)[0]
    return float(ys.min()) - 2 if len(ys) > 10 else None

def next_gap_top(rgb):
    """The gap top of the first pipe that reaches the bird or is ahead of it:
    the first run of sky rows below y=76 in a column of that pipe."""
    img = rgb.astype(int)
    green = (img[..., 1] > 120) & (img[..., 1] > img[..., 0] + 30) & (img[..., 2] < 120)
    sky = (img[..., 2] > 170) & (img[..., 1] > 170) & (img[..., 0] < 150)
    is_pipe = green[:76].sum(axis=0) > 30           # columns inside a top pipe
    for x in range(BIRD_X0, rgb.shape[1] - 3):
        if is_pipe[x]:
            # Measure at the pipe's right edge, clear of the birds.
            right = x
            while right + 1 < len(is_pipe) and is_pipe[right + 1]:
                right += 1
            col = sky[:GROUND_Y, max(right - 3, x)]
            for y in range(76, 240):
                if col[y:y + 6].all():
                    return y
            return None
    return None

async def main(url, out, seconds):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.csv", "w")
    async with aiohttp.ClientSession() as s, s.ws_connect(url) as ws:
        await ws.send_json({"t": "join", "mode": "race"})
        t0, n, phase, prev_top, last, gap = time.time(), 0, 0, None, 0.0, None
        async for msg in ws:
            if time.time() - t0 > seconds:
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                d = json.loads(msg.data)
                if d.get("t") == "me" and d.get("me"):
                    me = d["me"]
                    if me["phase"] != phase:
                        log.write(f"# frame {n} phase {me['phase']} you {me['you_score']} ai {me['ai_score']}\n")
                    phase = me["phase"]
                    if phase in (1, 3) and time.time() - last > 1.0:
                        await ws.send_json({"t": "restart" if phase == 3 else "flap"}); last = time.time()
                continue
            (out / f"f{n:05d}.jpg").write_bytes(msg.data); n += 1
            if phase != 2:
                continue
            rgb = np.asarray(Image.open(io.BytesIO(msg.data)).convert("RGB"))
            top, found = bird_top(rgb), next_gap_top(rgb)
            gap = found if found is not None else gap
            vy = top - prev_top if top is not None and prev_top is not None else 0.0
            limit = (gap + 100 - BIRD_H - 14) if gap is not None else 220
            high = gap is not None and top is not None and top < gap + 35
            if top is not None and top > limit and vy >= 0 and not high and time.time() - last > 0.1:
                await ws.send_json({"t": "flap"}); last = time.time()
            log.write(f"{n - 1},{top},{gap},{limit}\n")
            prev_top = top
    print("frames", n)

asyncio.run(main(*sys.argv[1:3], float(sys.argv[3])))
