// Headless smoke test: serves dist/ (or tests APP_URL, e.g. a running Vite dev server),
// opens it in Chromium with WebGPU, lets the fly play for a while, and reports
// console errors, HUD stats and a screenshot. Runs inside Docker (see README).
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import pw from "playwright-core";

const { chromium } = pw;

const root = process.env.DIST ?? "/app/dist";
const out = process.env.OUT ?? "/out";
const seconds = Number(process.env.SECONDS ?? 20);
const adapter = process.env.ADAPTER ?? "swiftshader";

const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm" };
// BASE_PATH=/repo/ serves dist below a subpath, like GitHub Pages project sites.
const basePath = process.env.BASE_PATH ?? "/";
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x").pathname;
  if (!url.startsWith(basePath)) {
    res.writeHead(404);
    return res.end();
  }
  const path = normalize(decodeURIComponent(url.slice(basePath.length)) || "index.html").replace(/^\/+/, "");
  const file = join(root, path || "index.html");
  try {
    if (!(await stat(file)).isFile()) throw new Error();
    res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" });
    res.end(await readFile(file));
  } catch {
    res.writeHead(404);
    res.end();
  }
});
if (!process.env.APP_URL) await new Promise((r) => server.listen(4173, "127.0.0.1", r));

const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"];
if (adapter === "swiftshader") {
  args.push("--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader", "--enable-unsafe-swiftshader");
} else {
  args.push("--use-angle=vulkan");
}
// The full Chromium build in new-headless mode (the headless shell has no WebGPU on Linux).
const browser = await chromium.launch(
  process.env.CHROMIUM ? { headless: true, executablePath: process.env.CHROMIUM, args: [...args, "--headless=new", "--no-sandbox"] } : { headless: true, channel: "chromium", args },
);
const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

await page.goto(`${process.env.APP_URL ?? `http://127.0.0.1:4173${basePath}`}${process.env.QUERY ?? ""}`);
const t0 = Date.now();
await page.waitForSelector("#start:not([hidden]), #loadError:not([hidden])", { timeout: 180_000 });
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s:`, await page.textContent("#loadLabel"));
if (await page.isVisible("#loadError")) {
  console.log("LOAD ERROR:", await page.textContent("#loadError"));
} else {
  await page.click("#start");
  for (let s = 0; s < seconds; s++) {
    await page.waitForTimeout(1000);
    const line = await page.evaluate(() => {
      const { game, rates } = window.aimbug;
      const deg = (r) => Math.round((r * 180) / Math.PI);
      const t = game.targets.find((x) => x.alive);
      const lr = (name) => `${rates.get(name, 1).toFixed(0)}/${rates.get(name, 2).toFixed(0)}`;
      return (
        `brain ${(game.time / 1000).toFixed(1)}s heading ${deg(game.heading)} pitch ${deg(game.pitch)} target az ${t ? deg(game.relAz(t)) : "-"} el ${t ? deg(game.relEl(t)) : "-"} | ` +
        `DNa02 ${lr("DNa02")} DNp53 ${lr("DNp53")} LC4 ${lr("LC4")} LPLC2 ${lr("LPLC2")} pIP10 ${lr("pIP10")} GF ${lr("DNp01")} | ` +
        `shots ${game.stats.shots} hits ${game.stats.hits} | ${document.querySelector("#brainstats")?.textContent.split("\n")[0]}`
      );
    });
    if (!process.env.QUIET || s % 10 === 9) console.log(`t+${s + 1}s ${line}`);
    const dumpAt = (process.env.DUMP_AT ?? "").split(",").filter(Boolean).map(Number);
    const spikesNow = Number((line.match(/([\d,]+) spikes\/s/)?.[1] ?? "0").replace(/,/g, ""));
    if (spikesNow > 80000 && !globalThis.dumpedIgnition) {
      globalThis.dumpedIgnition = true;
      console.log(`!!! IGNITION at t+${s + 1}s: ${line}`);
      dumpAt.push(s + 1);
    }
    if (dumpAt.includes(s + 1)) {
      console.log(`--- most active types at t+${s + 1}s ---\n` + (await page.evaluate(() => window.aimbug.topActive(25))).join("\n"));
    }
  }
  const summary = await page.evaluate(() => {
    const { stats, time } = window.aimbug.game;
    return {
      brainSeconds: +(time / 1000).toFixed(1),
      kills: stats.kills,
      score: stats.score,
      shots: stats.shots,
      killsPerMin: +((stats.kills * 60000) / time).toFixed(1),
      accuracy: stats.shots ? +((100 * stats.hits) / stats.shots).toFixed(1) : 0,
      onTargetPct: +((100 * stats.onTargetMs) / stats.playMs).toFixed(1),
      shotsPerSec: +((stats.shots * 1000) / time).toFixed(2),
      seizures: window.aimbug.seizures(),
    };
  });
  console.log("SUMMARY", process.env.QUERY ?? "", JSON.stringify(summary));
  const bins = await page.evaluate(() =>
    window.aimbug.songBins.map((b) => ({
      offsetBelowDeg: b.maxDeg,
      samples: b.samples,
      meanSongHz: +(b.rateSum / Math.max(1, b.samples)).toFixed(1),
      pctOverThreshold: +((100 * b.overThreshold) / Math.max(1, b.samples)).toFixed(1),
    })),
  );
  console.log("SONG", JSON.stringify(bins));
  const samples = await page.evaluate(() => window.aimbug.samples ?? null);
  if (samples) {
    const { writeFile } = await import("node:fs/promises");
    await writeFile(`${out}/samples.json`, JSON.stringify(samples));
    console.log(`wrote ${samples.length} samples`);
  }
  if (process.env.POPUP) {
    // fake a hit to check the score popup
    await page.evaluate(() => {
      const g = window.aimbug.game;
      g.onHit(g.targets[0], [{ points: 100, label: "Kill" }, { points: 50, label: "Quick Kill" }, { points: 50, label: "Double Kill" }]);
    });
    await page.waitForTimeout(250);
    await page.screenshot({ path: `${out}/popup.png` });
  }
  if (process.env.INTERACT) {
    // open two knob explanations and drag the brain view
    const infos = await page.$$(".knob .info");
    for (const i of infos.slice(0, 2)) await i.click();
    const box = await (await page.$("#brain")).boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 90, box.y + box.height / 2 + 40, { steps: 10 });
    await page.mouse.up();
  }
  // Prefer a frame with a live target inside the 100 deg view.
  await page
    .waitForFunction(
      () => {
        const g = window.aimbug.game;
        return g.targets.some((t) => t.alive && Math.abs(g.relAz(t)) < 0.6 && Math.abs(g.relEl(t)) < 0.4);
      },
      null,
      { timeout: 15000, polling: 50 },
    )
    .catch(() => console.log("(no target in view for the screenshot)"));
  await page.screenshot({ path: `${out}/smoke.png` });
  const brainEl = await page.$("#brain");
  await brainEl.scrollIntoViewIfNeeded();
  await brainEl.screenshot({ path: `${out}/brain.png` });
}
console.log("--- console ---\n" + logs.slice(0, 60).join("\n"));
await browser.close();
if (server.listening) server.close();
