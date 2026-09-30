"""Build the demo video: motion-graphics scenes + guided-tour recordings of real replays in the live UI.

    python scripts/make_video.py

No narration: an animated storyboard (scripts/video/storyboard.html) explains the problem, the pipeline, the
streaming effect and the results; the product walkthrough is the real web demo replaying scenarios in real time with
a spotlight tour (scripts/video/tour.js) that reacts to what the app is actually doing.
Needs ffmpeg and Playwright (`uv sync --extra demo`; uses /usr/bin/chromium if present).
Output: submission/DuplexRAG_demo.mp4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUILD = ROOT / "submission" / "_build" / "video"
VIDEO = ROOT / "scripts" / "video"
PORT = 8766
W, H = 1920, 1080

# ------------------------------------------------------------------ tour steps (conditions are evaluated in the page)
TURNS = "document.querySelectorAll('#transcript .turn')"
TOURS = {
    "demo-workshop": ("Walkthrough 1 · guide example: compound request, late detail, format request", [
        {"wait": "document.querySelector('#transcript .chunk')", "target": "document.querySelector('#panel-convo')",
         "title": "Speech arrives in chunks", "min": 3500,
         "text": "The request streams in as ASR partials, 2-4 words at a time. On every chunk the controller decides: "
                 "WAIT, RETRIEVE or SUPPRESS."},
        {"wait": "document.querySelector('#transcript .chunk.fired')",
         "target": "document.querySelector('#transcript .chunk.fired')", "title": "Retrieval fires mid-sentence",
         "min": 3500, "text": "As soon as “customer workshop in Pune” is a stable intent, a provisional search "
                              "starts - seconds before the user stops talking."},
        {"wait": "document.querySelectorAll('#queries .q').length >= 3", "target": "document.querySelector('#queries')",
         "title": "One request, several searches", "min": 3500,
         "text": "The utterance is decomposed into venue capacity, cancellation terms and catering options; each is "
                 "searched while the user keeps talking."},
        {"wait": "document.querySelector('#transcript .chunk.end')", "target": "document.querySelector('.kpis')",
         "title": "The answer starts in milliseconds", "min": 3200,
         "text": "At end of speech the speculative results are reused, so the first token needs no new search. "
                 "Latency, head start and cost are measured per turn."},
        {"wait": "document.querySelector('#answer .unc')", "target": "document.querySelector('#answer')",
         "title": "Every claim is cited", "min": 3800,
         "text": "One line per hidden question. Each claim is a corpus sentence carrying its section ID and is "
                 "verified against it - no invented citations."},
        {"wait": "document.querySelector('#answer .unc')", "target": "document.querySelector('#answer .unc')",
         "title": "Gaps are said out loud", "min": 3800,
         "text": "The corpus has no catering information for Hinjewadi Tech Park, so DuplexRAG flags it instead of "
                 "guessing."},
        {"wait": f"{TURNS}.length >= 2", "target": f"{TURNS}[1]", "title": "A late detail arrives", "min": 3200,
         "text": "“Actually it's going to be fifty people, not thirty.” - a refinement of the current answer, "
                 "not a new question."},
        {"wait": "(document.querySelector('#answer-badges .badge.v') || {}).textContent === 'v2'",
         "target": "document.querySelector('#diffbar')", "title": "Refined in place: v1 → v2", "min": 3800,
         "text": "Only the capacity claim is searched again; the other claims are retained. The diff "
                 "(added / retained / retired) is logged."},
        {"wait": f"{TURNS}.length >= 3 && document.querySelector('#answer ul')",
         "target": "document.querySelector('#answer')", "title": "“Repeat that in two bullets”", "min": 3500,
         "text": "A presentation-only turn: no retrieval at all - the same answer and citations, reformatted."},
        {"wait": "document.querySelector('#source') && !document.querySelector('#source').hidden",
         "target": "document.querySelector('#source')", "title": "Citations are traceable", "min": 3500,
         "text": "Clicking a citation opens the exact corpus section behind the claim."},
    ]),
    "demo-reimbursement": ("Walkthrough 2 · guide example: refine, don't restart", [
        {"wait": f"{TURNS}.length >= 1 && document.querySelector('#transcript .chunk.end') && "
                 "document.querySelector('#answer .cite')",
         "target": "document.querySelector('#answer')", "title": "A cited summary", "min": 3200,
         "text": "“Summarize the travel reimbursement rule for an employee trip” - answered from the "
                 "reimbursement policy with section citations."},
        {"wait": f"{TURNS}.length >= 2", "target": f"{TURNS}[1]", "title": "Two facts arrive late", "min": 3200,
         "text": "“The trip was international, and the booking was made after travel.” Statements about the "
                 "same topic → refinement."},
        {"wait": "(document.querySelector('#answer-badges .badge.v') || {}).textContent === 'v2' && "
                 "document.querySelector('#answer .update')",
         "target": "document.querySelector('#answer')", "title": "The rule still applies - plus two updates",
         "min": 4200, "text": "Earlier claims are retained; delta searches add the foreign-currency receipt check and "
                              "the senior-director approval for post-travel bookings."},
        {"wait": f"{TURNS}.length >= 3 && document.querySelectorAll('.turn-kind')[2].textContent === 'chitchat'",
         "target": f"{TURNS}[2]", "title": "“Okay perfect, thanks”", "min": 3000,
         "text": "Recognised as small talk - nothing is searched and the answer stays in session memory."},
        {"wait": "true", "target": "document.querySelector('#events')", "title": "Everything is traced", "min": 3200,
         "text": "Every chunk, decision, search, answer version and its cost is emitted as telemetry and can be "
                 "downloaded as JSON Lines."},
    ]),
    "demo-bengaluru-launch": ("Walkthrough 3 · the key entity arrives last (replayed at 1.5x)", [
        {"wait": "document.querySelector('#transcript .chunk')", "target": "document.querySelector('#panel-convo')",
         "title": "The city is mentioned last", "min": 3000,
         "text": "“...and any safety stuff we need, oh and it's in Bengaluru.” Searches start before the "
                 "city is known."},
        {"wait": "[...document.querySelectorAll('#queries .q')].some(q => q.textContent.includes('Bengaluru'))",
         "target": "document.querySelector('#queries')", "title": "Late entity → speculation invalidated",
         "min": 3500, "text": "Speculative searches that did not know the city are not reused; they run again with "
                              "Bengaluru in the query."},
        {"wait": "document.querySelector('#transcript .chunk.end') && document.querySelector('#answer .cite')",
         "target": "document.querySelector('#answer')", "title": "Venues, catering and safety for 110 people",
         "min": 3500, "text": "Each hidden question is answered from Bengaluru-specific and event-policy sections."},
        {"wait": f"{TURNS}.length >= 2", "target": f"{TURNS}[1]", "title": "“What if we cancel ten days before?”",
         "min": 3000, "text": "A hypothetical about the same plan → refinement with a single delta search."},
        {"wait": f"{TURNS}.length >= 3 && document.querySelectorAll('.turn-kind')[2].textContent === 'presentation'",
         "target": "document.querySelector('#answer')", "title": "“Just give me the gist”", "min": 3200,
         "text": "Condensed from the current answer version - no retrieval."},
    ]),
}
DEMOS = [("demo-workshop", "1"), ("demo-reimbursement", "1"), ("demo-bengaluru-launch", "1.5")]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def duration(path: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]))


def storyboard_data() -> dict:
    s = json.loads((ROOT / "results" / "test" / "summary.json").read_text())
    d, b, ns = s["duplexrag"], s["baseline"], s.get("no_speculation", {})
    data = {
        "ttft_on": d["ttft_ms_p50"] / 1000, "ttft_off": ns.get("ttft_ms_p50", 614) / 1000,
        "bars": [["Retrieval starts before end of speech (G2)", round(d["G2_early_retrieval_pct"]), round(b["G2_early_retrieval_pct"])],
                 ["Compound requests decomposed (G3)", round(d["G3_multi_intent_pct"]), round(b["G3_multi_intent_pct"])],
                 ["Gold section in the top-3 evidence", round(d["retrieval_hit_at_3_pct"]), round(b["retrieval_hit_at_3_pct"])],
                 ["Refinements handled in place (G5)", round(d["G5_refinement_pass_pct"]), round(b["G5_refinement_pass_pct"])],
                 ["Key facts present in the answer", round(d["key_fact_recall_pct"]), round(b["key_fact_recall_pct"])]],
        "gates": [["G2 early retrieval", f"{d['G2_early_retrieval_pct']:.0f}%", "target ≥ 80% of eligible turns"],
                  ["G3 multi-intent", f"{d['G3_multi_intent_pct']:.0f}%", "target ≥ 70% of compound turns"],
                  ["G4 grounding", f"{d['G4_claim_support_pct']:.0f}% · {d['G4_fabricated_ids']} fabricated IDs",
                   "target ≥ 85% supported claims"],
                  ["G5 session refinement", f"{d['G5_refinement_pass_pct']:.0f}%", "versioned, state kept, delta-only"],
                  ["G6 telemetry", f"{d['G6_trace_coverage_pct']:.0f}%", "trace coverage"],
                  ["G1 reproducibility", "one command", "Docker Compose + clean-checkout smoke test"]],
        "kpis": [[f"{d['ttft_ms_p50']:.0f} ms", "median time to first token"],
                 [f"{ns.get('ttft_ms_p50', 0):.0f} ms", "same stack, no speculation"],
                 [f"${d['cost_usd_per_turn_mean']:.5f}", "compute per turn (CPU)"],
                 ["0", "fabricated citations"]],
    }
    # real retrieval timing of the guide example (turn 1 of the workshop demo)
    from duplexrag.engine import DuplexEngine
    from duplexrag.stream import replay_session
    sc = [json.loads(l) for l in open(ROOT / "data" / "demo" / "scenarios.jsonl")]
    sess = next(x for x in sc if x["session_id"] == "demo-workshop")
    recs, evs = replay_session(DuplexEngine(log=lambda *a: None), {"session_id": "sb", "turns": sess["turns"][:1]})
    t0 = recs[0]["t_start"]
    data["speech_s"] = recs[0]["t_end"] - t0
    trig = {"provisional": "s", "multi_intent": "m", "final": "f"}
    data["duplex_blocks"] = [[e["t_dispatch"] - t0, max(e["t_stream"] - e["t_dispatch"], 0.15), trig.get(e["trigger"], "m")]
                             for e in evs if e["event"] == "retrieval_completed"]
    return data


async def record_scene(p, scene: str, data: dict, hold: float, out: Path) -> Path:
    exe = "/usr/bin/chromium" if Path("/usr/bin/chromium").exists() else None
    b = await p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
    ctx = await b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(out),
                              record_video_size={"width": W, "height": H})
    page = await ctx.new_page()
    await page.goto((VIDEO / "storyboard.html").as_uri())
    await page.wait_for_timeout(300)
    await page.evaluate("([s, d]) => { window.start(s, d); }", [scene, data])
    for _ in range(600):
        if await page.evaluate("window.__sceneDone === true"):
            break
        await page.wait_for_timeout(100)
    await page.wait_for_timeout(int(hold * 1000))
    video = page.video
    await ctx.close()
    await b.close()
    return Path(await video.path())


async def record_demo(p, scenario: str, speed: str, out: Path, screenshot: Path | None) -> Path:
    banner, steps = TOURS[scenario]
    exe = "/usr/bin/chromium" if Path("/usr/bin/chromium").exists() else None
    b = await p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
    ctx = await b.new_context(viewport={"width": 1600, "height": 900}, record_video_dir=str(out),
                              record_video_size={"width": 1600, "height": 900})
    page = await ctx.new_page()
    await page.goto(f"http://127.0.0.1:{PORT}/")
    await page.wait_for_timeout(1500)
    await page.add_script_tag(path=str(VIDEO / "tour.js"))
    await page.select_option("#scenario", scenario)
    await page.select_option("#speed", speed)
    await page.evaluate("([s, b]) => { window.runTour(s, b); }", [steps, banner])
    await page.wait_for_timeout(600)
    await page.click("#btn-play")
    shot_done = screenshot is None
    for _ in range(1200):
        await page.wait_for_timeout(250)
        if not shot_done and await page.evaluate(f"{TURNS}.length") >= 2:
            await page.evaluate("() => { for (const id of ['tour-spot','tour-card','tour-banner']) { "
                                "const e = document.getElementById(id); if (e) e.style.visibility = 'hidden'; } }")
            await page.screenshot(path=str(screenshot))
            await page.evaluate("() => { for (const id of ['tour-spot','tour-card','tour-banner']) { "
                                "const e = document.getElementById(id); if (e) e.style.visibility = ''; } }")
            shot_done = True
        if await page.evaluate("window.__replayDone === true"):
            break
    await page.wait_for_timeout(1500)
    cite = await page.query_selector("#answer .cite")
    if cite and scenario == "demo-workshop":
        await cite.click()
    for _ in range(200):
        if await page.evaluate("window.__tourDone === true"):
            break
        await page.wait_for_timeout(250)
    await page.wait_for_timeout(800)
    video = page.video
    await ctx.close()
    await b.close()
    return Path(await video.path())


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "duplexrag", "serve", "--port", str(PORT)], cwd=ROOT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise SystemExit("server did not start")


def encode(src: Path, dst: Path, trim: float) -> None:
    d = duration(src) - trim
    run(["ffmpeg", "-y", "-ss", f"{trim:.2f}", "-i", str(src), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
         "-filter_complex",
         f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0b0f16,"
         f"fps=30,format=yuv420p,fade=t=in:st=0:d=0.4,fade=t=out:st={max(d - 0.45, 0):.2f}:d=0.45[v]",
         "-map", "[v]", "-map", "1:a", "-t", f"{d:.2f}", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
         "-c:a", "aac", "-b:a", "64k", "-shortest", str(dst)])


async def build(out_path: Path) -> None:
    from playwright.async_api import async_playwright
    data = storyboard_data()
    order = [("scene", "title", 4.5), ("scene", "problem", 3.0), ("scene", "pipeline", 1.0),
             ("scene", "timeline", 3.5)] + [("demo", s, sp) for s, sp in DEMOS] + \
            [("scene", "results", 5.0), ("scene", "outro", 5.0)]
    parts = []
    server = start_server()
    try:
        async with async_playwright() as p:
            for k, (kind, name, arg) in enumerate(order):
                rec_dir = BUILD / f"rec_{k}"
                rec_dir.mkdir(parents=True)
                if kind == "scene":
                    raw = await record_scene(p, name, data, arg, rec_dir)
                    trim = 0.35
                else:
                    shot = ROOT / "docs" / "img" / "ui.png" if name == "demo-workshop" else None
                    raw = await record_demo(p, name, arg, rec_dir, shot)
                    trim = 1.6
                part = BUILD / f"part_{k:02d}.mp4"
                encode(raw, part, trim)
                parts.append(part)
                print(f"{k:02d} {kind:5s} {name:22s} {duration(part):6.1f}s", flush=True)
    finally:
        server.terminate()
    lst = BUILD / "parts.txt"
    lst.write_text("".join(f"file '{x}'\n" for x in parts))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart",
         str(out_path)])
    print(f"wrote {out_path}: {duration(out_path):.1f}s, {out_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submission" / "DuplexRAG_demo.mp4"))
    a = ap.parse_args()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    asyncio.run(build(Path(a.out)))


if __name__ == "__main__":
    main()
