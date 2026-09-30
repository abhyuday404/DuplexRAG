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

