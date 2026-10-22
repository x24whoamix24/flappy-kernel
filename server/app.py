# SPDX-License-Identifier: Apache-2.0
"""Web front end for the Flappy Kernel arcade.

The megakernel process (Mirage + co-tenants, COTENANT_MODE=4) and this server
share one block of memory (COTENANT_SHM). Players' flaps go into it; the GPU's
arcade block steps every game at 30 Hz and renders each player's frame into it.
This server encodes each new frame once and broadcasts it to everyone viewing
that game. LLM questions go through a queue directory that the megakernel
process serves one at a time, all in its one launch; the tokens it decodes are
read live from the shared block.

usage: python app.py --shm /flappy --layout layout.json --serve-dir /var/lib/flappy-kernel
"""
import argparse
import asyncio
import io
import json
import logging
import mmap
import os
import secrets
import struct
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web
from PIL import Image

log = logging.getLogger("arcade")
STATIC = Path(__file__).parent / "static"

PLAYER_MODES = {"race": 0, "solo": 1}
WATCH_MODE = 2
MAX_CLIENTS = 300           # open connections in total
MAX_CLIENTS_PER_IP = 8
MAX_MSGS_PER_S = 40         # a client above this is disconnected
SLOT_IDLE_S = 30            # a player slot without a flap for this long is freed
PROMPT_MAX_CHARS = 280
PROMPT_EVERY_S = 0          # seconds between questions per address; 0: no limit
QUEUE_MAX = 20
HISTORY = 6                 # finished answers shown to players
FIRST_PROMPT = "Explain what a GPU megakernel is"  # answered at startup, so the panel is never empty
STALL_S = 15                # no new frame for this long: the GPU process is down
JPEG_QUALITY = 80
INT32_MAX = 2**31 - 1


class Shared:
    """Typed access to the block the kernel shares with this process."""

    def __init__(self, shm_name, layout):
        self.L = layout
        path = "/dev/shm/" + shm_name.lstrip("/")
        while not os.path.exists(path) or os.path.getsize(path) < layout["size"]:
            time.sleep(0.5)
        fd = os.open(path, os.O_RDWR)
        self.m = mmap.mmap(fd, layout["size"])
        os.close(fd)
        self.w, self.h = layout["kW"], layout["kH"]
        self.slots = layout["slots"]

    def _slot(self, i, field):
        return self.L["slot"] + i * self.L["slot_size"] + self.L["slot_fields"][field]

    def i32(self, off):
        return struct.unpack_from("<i", self.m, off)[0]

    def i64(self, off):
        return struct.unpack_from("<q", self.m, off)[0]

    def slot_get(self, i, field):
        off = self._slot(i, field)
        return self.i64(off) if field == "frame_seq" else self.i32(off)

    def slot_set(self, i, field, v):
        struct.pack_into("<i", self.m, self._slot(i, field), v)

    def slot_bump(self, i, field):
        # Only this process writes these counters; the kernel compares them for
        # change, so wrapping is harmless.
        self.slot_set(i, field, (self.slot_get(i, field) + 1) & INT32_MAX)

    def frame(self, i):
        """A copy of slot i's latest frame (RGB), or None if it was being
        overwritten while copying."""
        seq = self.slot_get(i, "frame_seq")
        buf = self.slot_get(i, "frame_buf")
        size = self.w * self.h * 4
        off = self.L["slot_fb"] + (i * 2 + buf) * size
        rgba = np.frombuffer(self.m, dtype=np.uint8, count=size, offset=off).reshape(self.h, self.w, 4)
        rgb = np.array(rgba[:, :, :3])
        # The kernel publishes a frame's number only after drawing it, so once
        # the number has moved on, the frame after it may already be drawing
        # into the buffer just copied: any change means the copy may be torn.
        if self.slot_get(i, "frame_seq") != seq:
            return seq, None
        return seq, rgb

    def llm(self):
        launch = self.i64(self.L["llm_launch"])
        plen = self.i32(self.L["llm_prompt_len"])
        pos = self.i32(self.L["llm_pos"])
        n = max(0, min(pos + 1, self.L["llm_max_tokens"]) - max(plen, 0))
        if n == 0:
            return launch, []
        return launch, np.frombuffer(self.m, dtype=np.int64, count=n, offset=self.L["llm_tokens"] + 8 * plen).tolist()

    def counters(self):
        return {"updates": self.i64(self.L["updates"]), "episodes": self.i64(self.L["episodes"])}


def clean_prompt(text):
    """Collapses whitespace, caps the length and removes chat-template control
    tokens, so a question cannot impersonate the system or end the turn."""
    text = " ".join(str(text).split())
    # Repeat until none are left: removing one marker can join the characters
    # around it into another ("<<||" -> "<|").
    while "<|" in text or "|>" in text:
        text = text.replace("<|", "").replace("|>", "")
    return text[:PROMPT_MAX_CHARS]


class Client:
    def __init__(self, ws, ip):
        self.ws = ws
        self.ip = ip
        self.slot = None
        self.last_flap = 0.0
        self.busy = set()  # channels ("frame", "state", "notice") with a send in flight
        self.msg_times = []


class Arcade:
    def __init__(self, shared, serve_dir, tokenizer, question_interval=PROMPT_EVERY_S):
        self.sh = shared
        self.dir = Path(serve_dir)
        (self.dir / "queue").mkdir(parents=True, exist_ok=True)
        self.enqueue(FIRST_PROMPT)
        self.tok = tokenizer
        self.clients = set()
        self.owner = {}            # slot -> Client
        self.last_prompt = {}      # ip -> time of the last accepted question
        self.question_interval = question_interval  # seconds; 0 disables the limit
        self.best_human = 0
        self.frames = {}           # slot -> last frame_seq sent to its viewers
        self.state_json = "{}"
        self.history = []
        self.answers = deque(maxlen=HISTORY)
        self.last_done = ""        # newest done/ file already read
        self.rates = {"tok_s": 0.0, "upd_s": 0.0}
        self.health = (-1, time.time())
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.tasks = set()  # sends in flight; asyncio keeps only weak references
        # Slot 0 is the spectator view: the AI playing by itself.
        self.sh.slot_set(0, "mode", WATCH_MODE)
        self.sh.slot_set(0, "assigned", 1)
        self.sh.slot_bump(0, "start_seq")
        for i in range(1, self.sh.slots):
            self.sh.slot_set(i, "assigned", 0)

    # -- clients and slots --------------------------------------------
    def admit(self, ip):
        if len(self.clients) >= MAX_CLIENTS:
            return False
        return sum(c.ip == ip for c in self.clients) < MAX_CLIENTS_PER_IP

    def join(self, client, mode):
        if client.slot is None:
            if any(c.ip == client.ip and c.slot for c in self.clients):
                return None  # one game per address
            free = [i for i in range(1, self.sh.slots) if i not in self.owner]
            if not free:
                return None
            client.slot = free[0]
            self.owner[client.slot] = client
            self.frames.pop(client.slot, None)
        client.last_flap = time.time()
        self.sh.slot_set(client.slot, "mode", PLAYER_MODES[mode])
        self.sh.slot_bump(client.slot, "start_seq")
        self.sh.slot_set(client.slot, "assigned", 1)
        return client.slot

    def leave(self, client):
        if client.slot is not None and self.owner.get(client.slot) is client:
            self.sh.slot_set(client.slot, "assigned", 0)
            del self.owner[client.slot]
        client.slot = None

    # -- LLM -----------------------------------------------------------
    def queued(self):
        return sum(1 for p in (self.dir / "queue").iterdir() if p.suffix == ".json")

    def enqueue(self, text):
        """Adds a question to the queue the megakernel process serves."""
        name = f"{time.time():.6f}-{secrets.token_hex(3)}.json"
        tmp = self.dir / "queue" / (name + ".tmp")
        tmp.write_text(json.dumps({"prompt": text, "source": "player"}))
        os.replace(tmp, self.dir / "queue" / name)

    def queue_prompt(self, client, text):
        now = time.time()
        wait = self.question_interval - (now - self.last_prompt.get(client.ip, 0.0))
        if wait > 0:
            return f"One question every {self.question_interval:g} s. Try again in {int(wait) + 1} s."
        if self.queued() >= QUEUE_MAX:
            return "The question queue is full. Try again in a moment."
        self.enqueue(text)
        self.last_prompt = {ip: t for ip, t in self.last_prompt.items() if now - t < self.question_interval}
        self.last_prompt[client.ip] = now
        return None

    def read_llm(self):
        """Current answer and recent history; runs in a worker thread."""
        launch, ids = self.sh.llm()
        try:
            cur = json.loads((self.dir / "current.json").read_text())
        except (OSError, ValueError):
            cur = {}
        mine = isinstance(cur, dict) and cur.get("launch") == launch
        current = {"launch": launch, "source": "player" if mine else "",
                   "prompt": cur.get("prompt", "") if mine else "",
                   "text": self.tok.decode(ids, skip_special_tokens=True) if mine and ids else "",
                   "tokens": len(ids)}
        # Finished files never change: read each new one once, oldest first.
        done = self.dir / "done"
        names = sorted(p.name for p in done.iterdir()) if done.is_dir() else []
        for name in names:
            if name <= self.last_done:
                continue
            self.last_done = name
            try:
                d = json.loads((done / name).read_text())
                self.answers.append({
                    "launch": int(d["launch"]), "prompt": str(d.get("prompt", "")),
                    "text": str(d.get("text", "")),
                    "tok_s": float(d["generated"]) / max(float(d["seconds"]), 1e-6)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return current, list(self.answers)

    # -- background loops ------------------------------------------------
    async def frames_loop(self):
        """Encodes each new frame once and sends it to everyone viewing it."""
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(0.005)
            try:
                viewers = {}
                for c in self.clients:
                    viewers.setdefault(c.slot or 0, []).append(c)
                for slot, group in viewers.items():
                    seq = self.sh.slot_get(slot, "frame_seq")
                    if seq == self.frames.get(slot):
                        continue
                    seq, data = await loop.run_in_executor(self.pool, self.encode, slot)
                    if data is None:
                        continue
                    self.frames[slot] = seq
                    for c in group:
                        self.push(c, "frame", data)
            except Exception:
                log.exception("frame loop")

    def encode(self, slot):
        seq, rgb = self.sh.frame(slot)
        if rgb is None:
            return seq, None
        out = io.BytesIO()
        Image.fromarray(rgb).save(out, "JPEG", quality=JPEG_QUALITY)
        return seq, out.getvalue()

    def push(self, client, channel, *messages):
        """Sends messages on one of the client's channels. While a channel's last
        send is in flight new ones are dropped, so a slow client misses updates
        instead of queuing them."""
        if channel in client.busy:
            return
        client.busy.add(channel)
        task = asyncio.create_task(self.send(client, channel, messages))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def send(self, client, channel, messages):
        try:
            for m in messages:
                if isinstance(m, bytes):
                    await client.ws.send_bytes(m)
                else:
                    await client.ws.send_str(m)
        except (ConnectionError, RuntimeError):
            pass
        finally:
            client.busy.discard(channel)

    async def state_loop(self):
        """Live counters, the LLM feed and idle slots, four times a second."""
        loop = asyncio.get_running_loop()
        window = []  # (time, tokens decoded so far across launches, updates)
        launch_seen, tokens_before, tokens_last = 0, 0, 0
        while True:
            await asyncio.sleep(0.25)
            try:
                current, self.history = await loop.run_in_executor(self.pool, self.read_llm)
                now = time.time()
                counters = self.sh.counters()
                if current["launch"] != launch_seen:
                    tokens_before += tokens_last
                    launch_seen = current["launch"]
                tokens_last = current["tokens"]
                window.append((now, tokens_before + tokens_last, counters["updates"]))
                while now - window[0][0] > 3.0:
                    window.pop(0)
                dt = now - window[0][0]
                if dt > 0:
                    self.rates = {"tok_s": (window[-1][1] - window[0][1]) / dt,
                                  "upd_s": (window[-1][2] - window[0][2]) / dt}
                for slot, c in list(self.owner.items()):
                    self.best_human = max(self.best_human, self.sh.slot_get(slot, "you_score"))
                    if now - c.last_flap > SLOT_IDLE_S:
                        self.leave(c)
                        self.push(c, "notice", json.dumps({"t": "joined", "slot": None, "idle": True}))
                stats = {**counters, **self.rates, "players": len(self.owner),
                         "viewers": len(self.clients), "best_human": self.best_human,
                         "queued": self.queued(), "stalled": self.stalled()}
                self.state_json = json.dumps({"t": "state", "stats": stats, "llm": current,
                                              "history": self.history})
                for c in self.clients:
                    me = None
                    if c.slot:
                        me = {k: self.sh.slot_get(c.slot, k)
                              for k in ("phase", "you_score", "ai_score", "you_alive", "ai_alive", "mode")}
                    self.push(c, "state", json.dumps({"t": "me", "slot": c.slot, "me": me}), self.state_json)
            except Exception:
                log.exception("state loop")

    def stalled(self):
        seq, now = self.sh.slot_get(0, "frame_seq"), time.time()
        if seq != self.health[0]:
            self.health = (seq, now)
        return now - self.health[1] > STALL_S


def client_ip(request):
    """The client's address. Behind a trusted proxy, the address that proxy
    appended to X-Forwarded-For: the entries before it come from the client and
    can be anything."""
    if request.app["trust_proxy"]:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded.strip():
            return forwarded.split(",")[-1].strip()
    return request.remote or "?"


async def index(request):
    return web.FileResponse(STATIC / "index.html")


async def healthz(request):
    """200 while the GPU kernel is producing frames, 503 once it has stalled."""
    stalled = request.app["arcade"].stalled()
    return web.json_response({"stalled": stalled}, status=503 if stalled else 200)


async def websocket(request):
    arcade = request.app["arcade"]
    origin = request.headers.get("Origin")
    allowed = request.app["allowed_origins"]
    if origin and origin.split("://", 1)[-1] != request.host and origin not in allowed:
        raise web.HTTPForbidden(text="cross-origin connections are not allowed")
    ip = client_ip(request)
    if not arcade.admit(ip):
        raise web.HTTPServiceUnavailable(text="too many connections")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=2048, compress=False)
    await ws.prepare(request)
    client = Client(ws, ip)
    arcade.clients.add(client)
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            now = time.time()
            client.msg_times = [t for t in client.msg_times if now - t < 1.0] + [now]
            if len(client.msg_times) > MAX_MSGS_PER_S:
                await ws.close(code=1008, message=b"too many messages")
                break
            try:
                m = json.loads(msg.data)
            except ValueError:
                continue
            if not isinstance(m, dict):
                continue
            kind = m.get("t")
            if kind == "flap" and client.slot:
                client.last_flap = now
                arcade.sh.slot_bump(client.slot, "flap_seq")
            elif kind == "join" and m.get("mode") in PLAYER_MODES:
                slot = arcade.join(client, m["mode"])
                await ws.send_json({"t": "joined", "slot": slot})
            elif kind == "restart" and client.slot:
                client.last_flap = now
                arcade.sh.slot_bump(client.slot, "start_seq")
            elif kind == "leave":
                arcade.leave(client)
                await ws.send_json({"t": "joined", "slot": None})
            elif kind == "prompt":
                text = clean_prompt(m.get("text", ""))
                if text:
                    error = arcade.queue_prompt(client, text)
                    await ws.send_json({"t": "prompt", "ok": error is None, "error": error,
                                        "queued": arcade.queued()})
    finally:
        arcade.leave(client)
        arcade.clients.discard(client)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", default="/flappy")
    ap.add_argument("--layout", default=str(Path(__file__).parent / "layout.json"))
    ap.add_argument("--serve-dir", default="/var/lib/flappy-kernel")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--question-interval", type=float, default=PROMPT_EVERY_S,
                    help="seconds between questions from one address (default 0: no "
                         "limit; the queue still holds at most QUEUE_MAX questions)")
    ap.add_argument("--trust-proxy", action="store_true",
                    help="take client addresses from the last X-Forwarded-For entry "
                         "(only behind a proxy you control that appends to it)")
    ap.add_argument("--allow-origin", action="append", default=[],
                    help="extra origin allowed to open the WebSocket, e.g. https://arcade.example.com")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from transformers import AutoTokenizer

    layout = json.loads(Path(a.layout).read_text())
    log.info("waiting for the GPU process to create %s", a.shm)
    shared = Shared(a.shm, layout)
    app = web.Application()
    app["arcade"] = Arcade(shared, a.serve_dir, AutoTokenizer.from_pretrained(a.model),
                            a.question_interval)
    app["trust_proxy"] = a.trust_proxy
    app["allowed_origins"] = set(a.allow_origin)
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/healthz", healthz)

    async def start_loops(app):
        arcade = app["arcade"]
        app["loops"] = [asyncio.create_task(arcade.frames_loop()), asyncio.create_task(arcade.state_loop())]

    async def stop_loops(app):
        for t in app["loops"]:
            t.cancel()
        app["arcade"].pool.shutdown(wait=False)

    app.on_startup.append(start_loops)
    app.on_cleanup.append(stop_loops)
    web.run_app(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
