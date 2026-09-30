# Benchmark and evaluation report

All numbers are produced by `duplexrag bench` (discrete-event streaming replay, see docs/ARCHITECTURE.md §2) and can
be regenerated with `docker compose up` or `uv run duplexrag bench --split test --ablations`. Per-turn records,
telemetry traces and metrics for every configuration are in `results/<split>/<config>/`.

Hardware: laptop CPU (AMD Ryzen 5 4600H, 6 cores / 12 threads), 4 ONNX threads, no GPU; a web browser was running
during the measurements, so latency figures are conservative.

## Data and protocol

| Split | Sessions | Turns | Role |
|---|---|---|---|
| `dev.jsonl` | 18 | 43 | dev - used for all tuning |
| `devb.jsonl` | 32 | 77 | dev-b - first held-out, then used for diagnosis |
| `devc.jsonl` | 30 | 74 | dev-c - second held-out (engine v1), then used for diagnosis |
| `test.jsonl` | 30 | 74 | **test - final held-out** |

Every split was written by a separate AI agent that could read only the corpus (never the code, results or the
other splits). Utterances are spoken-style transcripts with gold sections, key facts and turn types; they are
streamed as ASR-like chunks of 2-4 words at 2.7 words/s with a 350 ms end-of-speech silence. The engine was tuned on
`dev` only. `devb` was our first held-out set; its results exposed generalisation bugs, so it became a diagnosis set
and a new held-out set was written after the fixes. That set (`devc`) was run once with the frozen engine ("engine
v1"; results archived in `results/archive/`) - it again revealed controller and uncertainty problems, so it too became
a diagnosis set, the engine was fixed and frozen again ("engine v2"), and the final `test` split was written
afterwards. We report every held-out run, including the ones that led to fixes.

## Evaluation gates (final held-out test)

| Gate | Target | DuplexRAG (held-out test) | Verdict | Baseline |
|---|---|---|---|---|
| G1 Reproducibility | pass/fail | one-command `docker compose up` (app + automated replay); `scripts/smoke_test.sh` passes from a clean `git archive` with the frozen lockfile | **pass**¹ | - |
| G2 Early retrieval | >= 80% of eligible turns | 100.0% early, 9.1% false triggers | **pass** | 0.0% |
| G3 Multi-intent | >= 70% of compound turns | 83.3% | **pass** | 12.5% |
| G4 Factual grounding | >= 85% claim support, 0 fabricated IDs | 100.0% of 332 claims, 0 fabricated | **pass** | 100.0% |
| G5 Session refinement | state continuity | 87.5% of refinement turns versioned, state kept, delta-only | **pass** | 0.0% |
| G6 Telemetry | 100% trace coverage | 100.0% | **pass** | - |

¹ The Docker daemon was not available on the build machine, so G1 was verified with the identical steps outside a
container: fresh `git archive` checkout → `uv sync --frozen` → model download → `duplexrag index` → streaming replay →
schema validation of the trace (`scripts/smoke_test.sh`).

## DuplexRAG vs the turn-based baseline (held-out test)

The baseline is a conventional turn-based RAG pipeline: it waits for the end of speech, retrieves once with the whole
utterance as the query (dense only, top 5), has no reranker, no decomposition and no session memory, and composes the
answer with the same extractive composer.

| Metric | duplexrag | baseline |
|---|---|---|
| Turn-type accuracy % | 97.3 | 63.5 |
| Retrieve / suppress decision accuracy % | 98.6 | 85.1 |
| G2 early retrieval % | 100.0 | 0.0 |
| ... started before the last chunk % | 93.7 | 0.0 |
| Median retrieval head start (ms before end of speech) | 3171.0 | - |
| G2 false triggers on no-retrieval turns % | 9.1 | 100.0 |
| G3 multi-intent % | 83.3 | 12.5 |
| Gold intents isolated per compound turn % | 89.6 | 53.5 |
| Retrieval hit@3 (per gold intent) % | 88.9 | 69.1 |
| Retrieval recall@5 % | 89.7 | 74.7 |
| Key-fact recall in answer % | 56.2 | 50.9 |
| Citation precision % | 58.0 | 44.1 |
| G4 claim support % | 100.0 | 100.0 |
| Fabricated citation IDs | 0 | 0 |
| Unanswerable needs flagged % | 10.0 | 10.0 |
| False 'not found' flags % | 1.9 | 0.0 |
| Turns with entity x aspect coverage notes % | 0.0 | 0.0 |
| G5 refinement pass % | 87.5 | 0.0 |
| G6 trace coverage % | 100.0 | - |
| TTFT p50 (ms after end of speech) | 7.0 | 10.7 |
| TTFT p95 (ms) | 308.6 | 17.2 |
| Retrieval CPU after end of speech, mean (ms) | 0.0 | 8.7 |
| CPU per turn, mean (ms) | 1450.0 | 10.9 |
| Model tokens per turn | 5915.6 | 24.8 |
| Cost per turn (USD) | 0.000072 | 0.000001 |

## Ablations (held-out test)

| Configuration | Turn type | G2 | False trig. | G3 | Hit@3 | Key facts | G5 | TTFT p50 ms | TTFT p95 ms | CPU ms/turn |
|---|---|---|---|---|---|---|---|---|---|---|
| `duplexrag` | 97.3 | 100.0 | 9.1 | 83.3 | 88.9 | 56.2 | 87.5 | 7.0 | 308.6 | 1450.0 |
| `dense_only` | 97.3 | 100.0 | 9.1 | 87.5 | 90.1 | 59.8 | 87.5 | 4.9 | 90.8 | 1166.7 |
| `bm25_only` | 97.3 | 100.0 | 9.1 | 66.7 | 77.8 | 53.6 | 87.5 | 5.4 | 149.3 | 1150.8 |
| `no_rerank` | 97.3 | 100.0 | 9.1 | 62.5 | 77.8 | 57.1 | 87.5 | 4.5 | 147.3 | 31.8 |
| `rule_controller` | 95.9 | 100.0 | 9.1 | 83.3 | 88.9 | 56.2 | 87.5 | 8.8 | 312.9 | 1448.2 |
| `model_controller` | 89.2 | 98.4 | 9.1 | 83.3 | 87.7 | 56.2 | 56.2 | 8.9 | 351.2 | 1442.5 |
| `no_decomposition` | 78.4 | 0.0 | 0.0 | 12.5 | 72.8 | 44.6 | 0.0 | 321.2 | 476.4 | 292.2 |
| `no_speculation` | 97.3 | 0.0 | 0.0 | 87.5 | 88.9 | 58.0 | 87.5 | 614.0 | 1701.8 | 585.7 |
| `baseline` | 63.5 | 0.0 | 100.0 | 12.5 | 69.1 | 50.9 | 0.0 | 10.7 | 17.2 | 10.9 |

* `dense_only` / `bm25_only`: one retriever instead of hybrid RRF (the cross-encoder still reranks).
* `no_rerank`: fused RRF order without the cross-encoder (and without cross-encoder sentence scoring).
* `rule_controller` / `model_controller`: the turn gate uses only the rules or only the trained classifier
  (default `duplexrag` = hybrid).
* `no_decomposition`: one query per utterance (same retrieval stack).
* `no_speculation`: identical pipeline, but retrieval waits for the end of speech - isolates the latency effect of
  streaming.

## All splits

(The dev, dev-b and dev-c runs overlapped with the demo-video recording on the same laptop, so their latency columns
are pessimistic; quality metrics are unaffected.)

| Split / engine | Turn type | G2 | False trig. | G3 | Hit@3 | Key facts | G4 | False 'not found' | G5 | TTFT p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| dev, engine v2 | 97.7 | 100.0 | 0.0 | 80.0 | 96.4 | 77.8 | 100.0 | 0.0 | 88.9 | 5.9 |
| dev-b, engine v2 | 93.5 | 95.5 | 0.0 | 70.4 | 80.2 | 55.5 | 100.0 | 0.0 | 87.5 | 257.7 |
| dev-c, engine v1 (its held-out run) | 81.1 | 89.2 | 44.4 | 84.0 | 81.3 | 49.5 | 100.0 | 29.1 | 50.0 | 275.6 |
| dev-c, engine v2 | 97.3 | 89.2 | 0.0 | 80.0 | 81.3 | 51.4 | 100.0 | 0.0 | 93.8 | 183.8 |
| **test, engine v2 (final)** | 97.3 | 100.0 | 9.1 | 83.3 | 88.9 | 56.2 | 100.0 | 1.9 | 87.5 | 7.0 |
| test, baseline | 63.5 | 0.0 | 100.0 | 12.5 | 69.1 | 50.9 | 100.0 | 0.0 | 0.0 | 10.7 |

## What the ablations show

* **Speculative (streaming) retrieval is what removes the wait.** `no_speculation` runs the identical pipeline but
  starts retrieving at end of speech: answer quality is unchanged, but median time-to-first-token grows from 7 ms to
  614 ms and p95 from 309 ms to 1.7 s. With speculation, retrieval for most sub-queries is already finished when the
  user stops; the remaining p95 comes from late entities that invalidate speculative results and from the last clause
  of long utterances.
* **Decomposition drives multi-intent quality.** One query per utterance (`no_decomposition`) drops G3 from 83% to
  12.5%, hit@3 from 89% to 73% and key-fact recall from 56% to 45%; it also disables refinements (G5 0%).
* **The cross-encoder is the main precision component.** Without it (`no_rerank`) hit@3 falls from 89% to 78% and
  G3 from 83% to 63%, while cost per turn falls about 36x (no reranker CPU). Key-fact recall is similar, because
  fused lists usually still contain the answering section in the top four.
* **Hybrid vs dense-only (architectural ablation 1).** On this 203-section synthetic corpus the bge-small dense
  retriever alone is marginally better than hybrid RRF (hit@3 90.1% vs 88.9%, key facts 59.8% vs 56.2%) and has a
  lower p95, while BM25 alone is clearly worse (hit@3 77.8%, G3 66.7%). BM25's contribution - exact tokens such as
  error codes, venue names and amounts - did not pay off on this corpus; we keep hybrid as the default because it is
  the safer choice for larger, more jargon-heavy corpora, and `DUPLEXRAG_RETRIEVAL_MODE=dense` is one variable away.
* **Rule-based vs model-based controller (architectural ablation 2).** The rule gate alone reaches 95.9% turn-type
  accuracy; the learned classifier alone (340 domain-general template utterances) reaches only 89.2% and handles far
  fewer refinements (G5 56% vs 88%); the hybrid default is best (97.3%). With little in-domain training data, explicit
  conversational cues generalise better than a small classifier; the classifier helps as an arbiter for cases the
  rules are unsure about.


## Edge-case failure analysis (final held-out test)

Five failure classes, each taken from the final run (`results/test/duplexrag/turns.jsonl`, traces in `trace.jsonl`).

### 1. Late details whose relevance needs world knowledge (G5 passes, answer misses the point)

*final-016 t2* - after asking about Cubbon Heritage Hall's terrace and caterers, the user adds "Oh, and sales wants a
DJ going till about eleven." The turn is correctly classified as a refinement and a single delta query is issued
(`sales wants a DJ going till 11 ... Cubbon Heritage Hall reception`), but the relevant rule - "no amplified music
after 10 pm" in the venue's *House Rules* section - shares no words with the utterance (DJ ↔ amplified music,
"till eleven" ↔ "after 10 pm"). The cross-encoder ranks the capacity and catering sections of the same venue higher
(best logit -0.24) and the answer repeats terrace facts. *final-018 t2* has the same shape: "I might be moving on in the
next year or two" should surface the L&D clawback clause (leaving within 12/24 months), but "moving on" never meets
"leaving the company" lexically, and the dense model does not bridge the euphemism either.
*Mitigation (next):* when a refinement targets an entity that is already in the answer, also score *every* section of
that entity against the delta (cheap: <10 sections), and add a small implication table or an optional verified LLM
query expansion for constraint-to-rule bridging.

### 2. A new constraint that opens a different aspect is anchored to the wrong intent

*final-013 t2* - the answer covers sign-off thresholds and whether the Sector 62 centre fits sixty people; the user
then says "the date just got pulled in, it's actually ten days from now". The correct delta is the 21-day venue
booking lead time and the Expedited Event Request form (Doc_01 §4). The delta sub-query is formed "in the context of
the closest existing intent", which here is the venue itself, so retrieval stays inside the venue's own document and
returns its cancellation and weekend-notice terms. The refinement is versioned and delta-only (so G5 counts it as a pass),
but the new aspect is missed. The same turn also shows a decomposition artefact: the spoken amount "four and a half
lakh" was split into two clauses ("total cost's 4", "half 100000"), producing two weak intents in the previous answer.
*Mitigation:* issue the delta query both with and without the nearest-intent context (the same hedging used for
first-turn sub-queries) and keep the better-scoring reading; treat "and a half" as part of a spoken number.

### 3. Unanswerable questions answered with related evidence

Only a minority of the unanswerable needs are flagged. *final-011 t2* "what about bereavement leave, like if a grandparent
passes away?" returns parental-leave sentences (best logit -7.2): the right *document family* but not an answer. We
removed the absolute reranker-score floor after the second held-out run because, on vague spoken phrasings, it flagged far
more answerable questions than unanswerable ones (29% false "not found" on dev-c before the change, 0-2% after). What is
left - typed-quantity checks and sibling-attribute checks - does not cover "does policy X exist?" questions.
*Mitigation:* a relative test (does any sentence of the evidence mention the focus term or a synonym of it?) applied only to
the head noun of existence questions ("is there any X", "what about X"), plus the optional LLM verifier.

### 4. False entity-attribute flag on a general-policy question

*final-005 t1* - "Which of our venues there is the biggest, and is there any extra safety or insurance stuff for a crowd
that size?" The safety requirements live in a general policy (Doc_12 §2), but the Bengaluru venue named earlier in the
utterance was carried into the sub-query, the venue's own overview section ranked first, and the entity-attribute check
concluded "Whitefield Innovation Hub does not mention safety" because *other* venues do mention safety. This is the one
false "not found" of the run.
*Mitigation:* only run the attribute check when the entity was named in the same clause (not carried from context), and
skip it when a document outside the entity family scores within a margin of the entity's own section.

### 5. Pronouns that refer to the *topic*, not an entity

*final-006 t1* - "I got back from a client trip three weeks ago and forgot to file my expenses. Is it too late now? And who
actually approves these things?" "these things" refers to expense claims; the anaphora rules carry the anchor clause's
salient terms ("expenses", "late") into the second question, which then retrieves the late-claims section again instead of
the approval workflow (Doc_22 §2).
*Mitigation:* for anaphora that replaces a noun phrase, carry only the noun ("expense claims"), not the anchor's
predicate terms ("late", "file").

### What went right on the same set

* Retrieval started before the end of speech on every retrieval-eligible turn (G2 100%). The single false trigger was a
  speculative search fired on "Sorry, someone was talking to me..." before "...can you repeat that?" arrived; the turn
  itself was then correctly handled as presentation-only. Two refinements were handled as new retrieval turns because
  they introduced several new terms ("what if the flat hunt drags on, like forty-five days?", "they've moved it to
  November eighth - is that one of our holidays?").
* Superseded-policy traps (the 2024 travel policy) were answered from the current policy in all three cases.
* Every claim in every answer was supported by its citation and no citation ID was fabricated (G4).

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

## Metric definitions

| Metric | Definition (implementation: `duplexrag/bench/metrics.py`) |
|---|---|
| Turn-type accuracy | engine turn kind (retrieval / refinement / presentation / chit-chat) equals the gold type (single, compound and unanswerable map to retrieval) |
| G2 early retrieval | share of retrieval-eligible turns whose first retrieval is dispatched before the end-of-speech event (a stricter variant: before the last ASR chunk arrives) |
| G2 false triggers | share of presentation / chit-chat turns on which any retrieval was dispatched (including speculative work) |
| G3 multi-intent | share of compound turns (>= 2 gold intents) where a bipartite matching between the engine's sub-queries and the answerable gold intents - an edge exists when a sub-query's top-3 evidence contains one of the intent's gold sections - covers at least two gold intents |
| Hit@3 / recall@5 | per answerable gold intent: any gold section among the top-3 evidence of the turn's current intents / share of gold sections among the top-5 |
| Key-fact recall | share of gold key facts (exact short strings from the gold sections, e.g. "INR 6,500") that appear in the answer text |
| Citation precision | share of cited sections that are gold for the turn (new-answer turns) |
| G4 claim support | share of answer claims whose citations all exist, were retrieved for this answer, contain every number of the claim and at least 70% of its content words; fabricated IDs = citations that do not exist in the corpus |
| Unanswerable flagged / false "not found" | share of turns with an unanswerable gold intent that carry an uncertainty flag / share of fully answerable turns with an intent-level "not found" flag (entity x aspect coverage notes are counted separately) |
| G5 refinement | gold refinement turns handled as refinements whose answer version is exactly parent + 1, whose diff retains or retires earlier claims (state continuity), and which issued at most four delta sub-queries none of which repeats a previous sub-query (no full re-search) |
| G6 trace coverage | share of turns whose trace contains every required event with stream/wall timestamps, version and cost fields (`scripts/check_trace.py`) |
| TTFT | stream time from the end-of-speech event to the first answer token; in replays retrieval jobs occupy a single virtual worker at their measured CPU time |
| Cost per turn | measured CPU time x USD 0.0447 per vCPU-hour x 4 threads (no LLM tokens in the default path) |

