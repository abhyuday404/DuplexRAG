"""Record the demo video: real replays in the live UI + deck slides + offline TTS narration + captions.

    python scripts/make_video.py --piper PATH/TO/piper --voice PATH/TO/en_US-lessac-medium.onnx

Needs: ffmpeg (with libass), Playwright (`uv sync --extra demo`, uses /usr/bin/chromium if present), the
deck PDF (scripts/make_deck.py) and a Piper voice. Output: submission/DuplexRAG_demo.mp4 (+ .srt).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "submission" / "_build" / "video"
TEAM = json.loads((ROOT / "submission" / "team.json").read_text())
DECK_PDF = ROOT / "submission" / f"{TEAM['college']}_{TEAM['team_name']}_Submission.pdf"
PORT = 8766
W, H = 1920, 1080


def res(key: str, cfg: str = "duplexrag", split: str = "test"):
    p = ROOT / "results" / split / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(cfg, {}).get(key)


def pct(key: str) -> str:
    v = res(key)
    return f"{v:.0f} percent" if v is not None else "most"


SEGMENTS = [
    {"kind": "slide", "page": 1, "text":
        "Hi, we are team Pokermons from V I T V. This is DuplexRAG, our submission for theme four of the Samsung "
        "PRISM Generative AI Hackathon: streaming live RAG."},
    {"kind": "slide", "page": 2, "text":
        "In a voice conversation people say one natural request, not a tidy query. One sentence can hide several "
        "questions, the key detail may arrive last, and users add details after they hear the answer. Waiting for "
        "the end of speech before searching leaves dead air, and restarting on every correction loses context."},
    {"kind": "slide", "page": 4, "text":
        "DuplexRAG is event driven. Every transcript chunk goes to a retrieval controller that decides whether to "
        "wait, retrieve, or suppress. A decomposer splits the utterance into the searches it implies and carries "
        "shared context into each. BM25 and dense retrieval are fused and reranked by a small cross encoder, on the "
        "CPU, while the user is still speaking. The session aware composer answers every intent with cited corpus "
        "sentences, verifies every claim, and flags what the corpus cannot support."},
    {"kind": "demo", "scenario": "demo-workshop", "screenshot": True, "text":
        "Here is the example from the theme guide, replayed in real time. The controller waits on the first words, "
        "fires a provisional search once Pune and thirty people are stable, and dispatches the cancellation and "
        "catering searches mid sentence. When the user stops, the answer streams almost at once: one line per "
        "hidden question, and a citation on every claim. Note the flag: catering for the Hinjewadi venue could not "
        "be verified from the corpus. Now a late detail: fifty people, not thirty. The answer becomes version two; "
        "only the capacity claim is searched again and the rest is kept. Finally, repeat that in two bullets: no "
        "retrieval at all, same citations, new format."},
    {"kind": "demo", "scenario": "demo-reimbursement", "text":
        "The second guide example. A travel reimbursement question gets a cited summary. Then the user adds that the "
        "trip was international and the booking was made after travel. DuplexRAG does not restart. It keeps the "
        "standard rule and adds the foreign currency receipt check and the senior director approval for post travel "
        "bookings. The thank you is recognised as small talk, so nothing is searched."},
    {"kind": "demo", "scenario": "demo-bengaluru-launch", "speed": "1.5", "text":
        "Here the key entity arrives last: oh, and it's in Bengaluru. Speculative searches that did not know the city "
        "are invalidated and run again, so the answer covers Bengaluru venues, catering and event safety for a "
        "hundred and ten people. A what if about cancelling ten days before refines the answer, and give me the gist "
        "condenses it without searching."},
    {"kind": "slide", "page": 8, "text":
        f"We measured everything with a discrete event replay on a held out set written independently after the "
        f"engine was frozen. Retrieval started before the end of speech in {pct('G2_early_retrieval_pct')} of "
        f"eligible turns. Compound requests were decomposed correctly in {pct('G3_multi_intent_pct')}. "
        f"{pct('G4_claim_support_pct').capitalize()} of claims were supported by their citation, with zero "
        f"fabricated document IDs, and {pct('G5_refinement_pass_pct')} of refinements were versioned and delta "
        f"only. The turn based baseline never retrieves early and cannot refine."},
    {"kind": "slide", "page": 10, "text":
        "DuplexRAG runs on a laptop CPU with two small open models, costs about a hundredth of a cent per turn, and "
        "ships with Docker, a replay benchmark, telemetry, and a live microphone demo. Thank you."},
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def duration(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).stdout
    return float(out.strip())


def tts(piper: str, voice: str, text: str, out: Path) -> float:
    subprocess.run([piper, "-m", voice, "-f", str(out), "--length-scale", "1.02", "--sentence-silence", "0.25"],
                   input=text, text=True, check=True, capture_output=True)
    return duration(out)


def start_server() -> subprocess.Popen:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "duplexrag", "serve", "--port", str(PORT)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise SystemExit("server did not start")


async def record(scenario: str, out_dir: Path, screenshot: Path | None, speed: str = "1") -> Path:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        exe = "/usr/bin/chromium" if Path("/usr/bin/chromium").exists() else None
        b = await p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        ctx = await b.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1,
                                  record_video_dir=str(out_dir), record_video_size={"width": 1600, "height": 900})
        page = await ctx.new_page()
        await page.goto(f"http://127.0.0.1:{PORT}/")
        await page.wait_for_timeout(1500)
        await page.select_option("#scenario", scenario)
        await page.select_option("#speed", speed)
        await page.wait_for_timeout(700)
        await page.click("#btn-play")
        shot_taken = screenshot is None
        for _ in range(900):
            await page.wait_for_timeout(250)
            if not shot_taken and await page.evaluate("document.querySelectorAll('.turn').length") >= 2:
                await page.screenshot(path=str(screenshot))
                shot_taken = True
            if await page.evaluate("window.__replayDone === true"):
                break
        await page.wait_for_timeout(1200)
        cite = await page.query_selector("#answer .cite")
        if cite:
            await cite.click()
            await page.wait_for_timeout(2600)
        video = page.video
        await ctx.close()
        await b.close()
        return Path(await video.path())


def srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    mnt, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(mnt):02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--piper", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission" / "DuplexRAG_demo.mp4"))
    a = ap.parse_args()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    if not DECK_PDF.exists():
        raise SystemExit(f"missing {DECK_PDF}; run scripts/make_deck.py first")
    run(["pdftoppm", "-r", "144", "-png", str(DECK_PDF), str(BUILD / "slide")])
    server = start_server()
    parts, subs, t0 = [], [], 0.0
    try:
        for k, seg in enumerate(SEGMENTS):
            wav = BUILD / f"narr_{k}.wav"
            nd = tts(a.piper, a.voice, seg["text"], wav)
            part = BUILD / f"part_{k}.mp4"
            if seg["kind"] == "slide":
                img = BUILD / f"slide-{seg['page']:02d}.png"
                dur = nd + 1.0
                run(["ffmpeg", "-y", "-loop", "1", "-i", str(img), "-i", str(wav), "-filter_complex",
                     f"[0:v]scale={W}:{H},fps=30,format=yuv420p[v];[1:a]adelay=400|400,apad[a]", "-map", "[v]",
                     "-map", "[a]", "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "medium", "-crf", "24",
                     "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(part)])
                lead = 0.4
            else:
                shot = ROOT / "docs" / "img" / "ui.png" if seg.get("screenshot") else None
                rec = asyncio.run(record(seg["scenario"], BUILD, shot, seg.get("speed", "1")))
                vd = duration(rec)
                dur = max(vd, nd + 1.5)
                run(["ffmpeg", "-y", "-i", str(rec), "-i", str(wav), "-filter_complex",
                     f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
                     f"fps=30,format=yuv420p,tpad=stop_mode=clone:stop_duration=30[v];[1:a]adelay=800|800,apad[a]",
                     "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "medium",
                     "-crf", "26", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(part)])
                lead = 0.8
            parts.append(part)
            # captions: split narration into sentences, time proportionally to their length
            sents = [x.strip() for x in re.split(r"(?<=[.!?:])\s+", seg["text"]) if x.strip()]
            total = sum(len(x) for x in sents)
            cur = t0 + lead
            for sent in sents:
                d = nd * len(sent) / total
                subs.append((cur, cur + d, sent))
                cur += d
            t0 += duration(part)
            print(f"segment {k}: {seg['kind']} {duration(part):.1f}s (narration {nd:.1f}s)", flush=True)
    finally:
        server.terminate()
    lst = BUILD / "parts.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    joined = BUILD / "joined.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
    srt = Path(a.out).with_suffix(".srt")
    srt.write_text("".join(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{txt}\n\n" for i, (s, e, txt) in enumerate(subs, 1)))
    style = ("FontName=DejaVu Sans,FontSize=11,PrimaryColour=&H00FFFFFF,BackColour=&H26101014,BorderStyle=4,"
             "Outline=0,Shadow=0,MarginV=10")
    run(["ffmpeg", "-y", "-i", str(joined), "-vf", f"subtitles={srt}:force_style='{style}'", "-c:v", "libx264",
         "-preset", "medium", "-crf", "26", "-c:a", "copy", str(a.out)])
    print(f"wrote {a.out}: {duration(Path(a.out)):.1f}s, {Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
