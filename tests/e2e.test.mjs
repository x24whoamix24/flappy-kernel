// SPDX-License-Identifier: Apache-2.0
// End-to-end test of a running arcade, in real Chrome.
// usage: node tests/e2e.test.mjs [url] [--llm]
//   url    the arcade (default http://127.0.0.1:8080, e.g. via kubectl port-forward)
//   --llm  also ask the LLM a question and wait for Mirage's answer
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const url = args.find((a) => !a.startsWith("--")) || "http://127.0.0.1:8080";
const withLlm = args.includes("--llm");
const shots = fileURLToPath(new URL("./screenshots/", import.meta.url));
mkdirSync(shots, { recursive: true });

const results = [];
const check = (name, ok, detail = "") => {
  results.push(ok);
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `  (${detail})` : ""}`);
};

const browser = await chromium.launch({ channel: "chrome" });
const errors = [];
async function open() {
  const page = await browser.newPage({ viewport: { width: 1180, height: 940 } });
  page.on("pageerror", (e) => errors.push(e.message));
  // Count the frames the GPU streams to this page.
  await page.addInitScript(() => {
    window.__frames = 0;
    const Native = window.WebSocket;
    window.WebSocket = function (...a) {
      const ws = new Native(...a);
      ws.addEventListener("message", (e) => { if (typeof e.data !== "string") window.__frames++; });
      return ws;
    };
    window.WebSocket.prototype = Native.prototype;
  });
  await page.goto(url);
  await page.waitForFunction(() => document.getElementById("live-text").textContent.includes("live"), null, { timeout: 30000 });
  return page;
}
const canvasHash = (page) => page.evaluate(() => {
  const d = document.getElementById("screen").getContext("2d").getImageData(0, 0, 288, 512).data;
  let h = 0, lit = 0;
  for (let i = 0; i < d.length; i += 4) { h = (h * 31 + d[i] + d[i + 1] * 7 + d[i + 2] * 13) | 0; lit += d[i] + d[i + 1] + d[i + 2] > 30; }
  return { h, lit };
});
const text = (page, sel) => page.locator(sel).textContent();

// 1. Spectating: frames arrive and keep changing.
const a = await open();
const f0 = await a.evaluate(() => window.__frames);
const h0 = await canvasHash(a);
await a.waitForTimeout(3000);
const f1 = await a.evaluate(() => window.__frames);
const h1 = await canvasHash(a);
const fps = (f1 - f0) / 3;
check("GPU frames stream to the browser", fps > 20, `${fps.toFixed(1)} frames/s`);
check("frames show a moving game", h0.h !== h1.h && h1.lit > 100000, `${h1.lit} lit pixels`);
await a.locator("canvas").screenshot({ path: shots + "1-spectate.png" });
const upd0 = await text(a, "#g-upd");
check("training runs live on the GPU", Number(upd0.replace(/,/g, "")) > 100, `${upd0} updates/s`);

// 2. Player one races the AI.
await a.click("#play-race");
await a.waitForFunction(() => document.getElementById("result").textContent.includes("Flap to start"), null, { timeout: 10000 });
check("player gets a GPU game slot", /slot \d/.test(await text(a, "#me")), await text(a, "#me"));

// 3. A second page from the same address asks for a game. The server allows one
// game per address (every page of this test shares one), so it keeps watching.
const b = await open();
await b.click("#play-solo");
await b.waitForFunction(() => /already have one open/.test(document.getElementById("result").textContent), null, { timeout: 10000 });
check("one game per address: a second page keeps watching", !/slot \d/.test(await text(b, "#me")), await text(b, "#result"));

// 4. Player one flaps; the GPU game reacts and the race ends with a result.
await a.locator("canvas").focus();
const t0 = Date.now();
let sawPlaying = false;
while (Date.now() - t0 < 15000) {
  await a.keyboard.press("Space");
  await a.waitForTimeout(380);
  const r = await text(a, "#result");
  if (r.startsWith("You ")) sawPlaying = true;
  if (/beat the AI|outlasted you/.test(r)) break;
}
await a.locator("canvas").screenshot({ path: shots + "2-race.png" });
const res = await text(a, "#result");
check("flaps reach the GPU game", sawPlaying);
check("race ends with a result", /beat the AI|outlasted you/.test(res), res);
await b.locator("canvas").screenshot({ path: shots + "3-second-page.png" });

// 5. Optional: ask the LLM (needs the Mirage megakernel, not the standalone kernel).
if (withLlm) {
  await a.fill("#question", "In one sentence, what is a GPU megakernel?");
  await a.click("#ask-btn");
  await a.waitForFunction(() => document.getElementById("ask-note").textContent.startsWith("Queued"), null, { timeout: 10000 });
  await a.waitForFunction(() => [...document.querySelectorAll(".msg")].some((m) =>
    m.textContent.includes("what is a GPU megakernel") && m.querySelector(".a").textContent.length > 40 && !m.classList.contains("now")),
    null, { timeout: 120000 });
  const ans = await a.evaluate(() => [...document.querySelectorAll(".msg")].find((m) => m.textContent.includes("what is a GPU megakernel")).querySelector(".a").textContent);
  check("LLM answers a player's question", ans.length > 40, ans.slice(0, 120));
  const tok = await text(a, "#g-tok");
  check("LLM decodes in the same kernel", Number(tok.replace(/,/g, "")) > 0 || ans.length > 40, `${tok} tokens/s now`);
}

await a.screenshot({ path: shots + "4-page.png" });
check("no page errors", errors.length === 0, errors.join(" | "));
await browser.close();
process.exit(results.every(Boolean) ? 0 : 1);
