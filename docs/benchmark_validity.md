## How the engine evolved (and why the held-out numbers are honest)

| Stage | What we learned | What changed |
|---|---|---|
| dev tuning | first end-to-end replays: sentence choice and uncertainty were the weak spots | cross-encoder sentence scoring during retrieval, typed-quantity checks, entity x aspect coverage |
| first held-out run (`devb`) | hit@3 72%, 36% false triggers, 47% false "not found" flags: rules over-fitted to dev phrasing (unknown words dropped, "eleven thirty" parsed as 41, "no sorry, I meant Chennai" lost the question) | unknown words kept as content, number/repair parsing, context hedging, follow-up speculation guard |
| second held-out run (`devc`, engine v1) | G2/G3/G4/G6 met, but 44% false triggers, 29% false "not found" and G5 50%: late details phrased as plain statements were treated as new questions; the absolute cross-encoder floor flagged vague but answerable requests | statements during an active answer = refinement; the absolute "not found" floor removed; attribute checks limited to questions |
| final held-out run (`test`, engine v2) | written after the freeze and run once | - |

Every rule change was motivated by a class of failure, not by a single utterance, and was checked for regressions on all
earlier splits (see "All splits" above).

## Threats to validity

* **Synthetic corpus and AI-authored benchmark.** The official corpus was not available to us; the corpus and all
  benchmark splits were written by AI agents (different agent instances from the one that wrote the code, with no
  access to it). Utterances are realistic but may be more regular than real users.
* **Small samples.** Each split has about 40-77 turns, so percentages have wide confidence intervals (about +/-10 points
  for gate-level rates, more for rates over the 7-16 refinement or presentation turns).
* **Simulated speech.** Chunks are clean transcript fragments at a fixed speaking rate; real ASR partials revise
  earlier words and contain recognition errors. The engine accepts cumulative (revisable) partials and the web demo
  uses a real streaming recogniser, but this is not benchmarked.
* **Strict key facts.** Key-fact recall uses exact substrings chosen by the benchmark authors; a correct answer that
  paraphrases or cites a different but valid section is counted as a miss, so this metric under-estimates answer quality.
* **Latency on a shared laptop.** Timings are real CPU measurements placed on a virtual worker; they depend on the
  machine and background load (a browser was running).
