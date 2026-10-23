# SPDX-License-Identifier: Apache-2.0
"""Records the arcade page during a race, for docs/media/race.gif: headless
Chrome on the same host as the server (so flaps arrive without lag) races the
AI, flying the bird from what the canvas shows. It first types a question into
the page and sends it, so the answer streams in during the race; after
<seconds> it stops flapping, so the recording ends on the result. Needs `pip install playwright` and
`python -m playwright install --with-deps chromium`.

    python bench/race_page.py http://localhost:8080 <video dir> <seconds>

The video (1280 x 720, the page as players see it) is a .webm in <video dir>;
ffmpeg turns it into the GIF."""
import sys, time
from playwright.sync_api import sync_playwright

READ = """() => {
  const d = document.getElementById('screen').getContext('2d').getImageData(0, 0, 288, 404).data;
  const px = (x, y) => { const i = (y * 288 + x) * 4; return [d[i], d[i + 1], d[i + 2]]; };
  const rows = (test) => { for (let y = 0; y < 404; y++) { let n = 0;
      for (let x = 51; x < 97; x++) if (test(...px(x, y))) n++;
      if (n >= 3) return y - 2; } return null; };
  const top = rows((r, g, b) => r > 200 && g > 150 && b < 120);
  const ai = rows((r, g, b) => b > 235 && r < 170 && g > 150);
  const green = (x, y) => { const [r, g, b] = px(x, y); return g > 120 && g > r + 30 && b < 120; };
  const sky = (x, y) => { const [r, g, b] = px(x, y); return b > 170 && g > 170 && r < 150; };
  const isPipe = (x) => { let n = 0; for (let y = 0; y < 76; y++) if (green(x, y)) n++; return n > 30; };
  let gap = null;
  for (let x = 57; x < 285 && gap === null; x++) {
    if (!isPipe(x)) continue;
    let right = x; while (right + 1 < 288 && isPipe(right + 1)) right++;
    const cx = Math.max(right - 3, x);
    for (let y = 76; y < 240; y++) { let ok = true;
      for (let k = 0; k < 6; k++) if (!sky(cx, y + k)) { ok = false; break; }
      if (ok) { gap = y; break; } }
    break;
  }
  return { top, ai, gap, result: document.getElementById('result').textContent };
}"""

QUESTION = ("Write a short story, about 400 words, about a little bird racing an AI "
            "through pipes inside a GPU.")

url, out, play_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
with sync_playwright() as p:
    b = p.chromium.launch()
    t_rec = time.time()  # the video starts about now; stage times below are relative to it
    stages = {}
    ctx = b.new_context(viewport={"width": 1280, "height": 720},
                        record_video_dir=out, record_video_size={"width": 1280, "height": 720})
    page = ctx.new_page()
    page.goto(url)
    page.wait_for_timeout(3000)
    page.click("#question")
    page.keyboard.type(QUESTION, delay=35)            # typed, as a player would
    page.wait_for_timeout(400)
    page.click("#ask-btn")
    stages["send"] = time.time() - t_rec
    page.wait_for_timeout(800)
    page.click("#play-race")
    page.wait_for_timeout(1200)
    page.locator("canvas").focus()
    page.keyboard.press("Space")                       # start the race
    stages["race"] = time.time() - t_rec
    t0, last, prev, gap = time.time(), 0.0, None, None
    while time.time() - t0 < play_s:
        s = page.evaluate(READ)
        gap = s["gap"] if s["gap"] is not None else gap
        top, ai = s["top"], s["ai"]
        vy = top - prev if top is not None and prev is not None else 0
        # A flap lifts the bird 45 px: flap between gap+50 (its peak stays
        # below the gap's top) and gap+62 (before it reaches the bottom),
        # as close to the AI's bird as that allows.
        limit = gap + 62 if gap is not None else 220
        if ai is not None:
            limit = min(limit, ai + 34)
        if gap is not None:
            limit = max(limit, gap + 50)
        if top is not None and top > limit and vy >= 0 and time.time() - last > 0.1:
            page.keyboard.press("Space"); last = time.time()
        prev = top
        page.wait_for_timeout(15)
    # Stop flapping: the bird falls and the page shows the result.
    for _ in range(200):
        if "again" in page.locator("#result").text_content():
            break
        page.wait_for_timeout(100)
    stages["result"] = time.time() - t_rec
    print("stages:", {k: round(v, 1) for k, v in stages.items()})
    print("result:", page.locator("#result").text_content())
    page.wait_for_timeout(3500)
    ctx.close(); b.close()
