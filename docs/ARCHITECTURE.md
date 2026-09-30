# DuplexRAG - System Architecture Brief

*Theme 4 - Streaming Live RAG · Team Pokermons (VITV)*

## 1. Problem and design goals

In a full-duplex voice conversation the user speaks one natural request, not a tidy query. A single
utterance can carry several needs ("a workshop venue in Pune for thirty people, the cancellation policy
and the catering options"), the key entity may arrive last ("...oh, and it's in Bengaluru"), details
arrive after the answer ("actually it's fifty people") and some turns need no retrieval at all
("repeat that in two bullets"). A turn-based RAG pipeline waits for the end of speech, searches once with
the whole utterance, restarts on every correction and searches even for formatting requests.

DuplexRAG is built around five goals taken from the brief:

1. **Latency hidden behind speech** - retrieval work should be done *while the user talks*, so the time to
   first token after end-of-speech is dominated by composition, not search.
2. **One utterance, several grounded answers** - decompose, retrieve per need, fuse, answer each need.
3. **Refine, don't restart** - late details update a versioned answer and fetch only the delta.
4. **Grounding by construction** - every claim cites a corpus chunk; missing evidence is said out loud.
5. **Parsimony** - no agent framework and no LLM in the default path; two small CPU models
   (33M bi-encoder, 22M cross-encoder), ~200 lines per stage, every component justified by telemetry.

## 2. Pipeline

```
 chunk(t) ─► [1] Controller ─► [2] Decomposer ─► [3] Retrieval & fusion ─► (speculative results, per sub-query)
                  │  wait / retrieve / suppress                                    │
 end-of-speech ───┴──────────────► finalise: reuse or fetch delta ─► [4] Synthesis ─► streamed answer + citations
                                                                                   └► [5] Telemetry (every step)
```

The engine is event-driven: `on_chunk()` for each ASR partial and `end_turn()` at end-of-speech.
Retrieval jobs run on a background worker, so chunk handling never blocks. The same engine runs
against a wall clock (web demo, live microphone) and against a discrete-event clock for replays: each
retrieval job really executes, its CPU time is measured and it is placed on a single virtual worker at
the stream time it was dispatched, which makes TTFT and early-retrieval numbers exact and reproducible.

### [1] Retrieval controller - *when* to retrieve

The controller has two layers.

**Turn gate** (what kind of turn is this?) - `retrieval` (new need), `refinement` (late constraint on the
current answer), `presentation` (reformat the current answer) or `chitchat`. Three interchangeable
implementations:

* *rule*: linguistic cues - presentation verbs referring back to the answer ("repeat that", "two
  bullets", "short version") with no new content; gratitude/acknowledgement without content;
  refinement cues ("actually", "hang on", "forgot to mention", hypotheticals such as "what if we
  cancel ten days before") or statement-only turns that restate a constraint; a follow-up *question*
  about something new stays a retrieval turn.
* *model*: softmax regression over the bge-small embedding plus 14 cue features (340 domain-general
  training utterances, trained in about a second, cached by content hash).
* *hybrid* (default): high-confidence rule decisions win; the classifier arbitrates the rest.

**Stability gate** (is there something worth searching yet?) - partial transcripts are segmented into
clauses; only clauses followed by a boundary are *complete*. A complete request clause with salient
content is dispatched immediately (trigger `multi_intent`); the in-progress tail clause may fire a
`provisional` search once it has enough salient content and does not end in a dangling word
("...in", "...and I need"). A clause re-fires only if it gains two or more terms or a new entity or
number ("Banyan" → "Banyan Court"), never on every token. Follow-up turns (an answer already exists)
wait for five words and new content terms, because they often start with "okay", "can you" or
"thanks" and are presentation or chit-chat. Presentation and chit-chat turns are *suppressed*: no
search at all.

### [2] Multi-intent decomposer - *what* to retrieve

* **Spoken-language normalisation**: fillers removed, spoken numbers converted ("a hundred and ten" →
  110, "eight oh nine" → 809, but "eleven thirty" is not 41), self-repairs applied ("what hotel, sorry,
  what flight class" → "what flight class"; "Bengaluru, no sorry, I meant Chennai" → "Chennai"),
  aliases ("the US" → USA, "pounds" → GBP).
* **Clause segmentation and labelling**: split at punctuation and coordinators; commas after function
  words are treated as pauses, not boundaries. Each clause is labelled *request* (an information need),
  *context* ("we're planning a launch for 110 people", "it's in Bengaluru"), *modifier* ("to Japan",
  attached to the previous request) or *filler*. Contentless follow-ups ("what am I supposed to do?")
  fold into the clause they refer to, while distinct questions ("...and how often must I change it?")
  stay separate needs.
* **Sub-query construction**: each request clause becomes one sub-query. Shared context is carried in:
  entities from context clauses and the anchor clause, topic terms of context clauses, the anchor's
  topic for anaphoric follow-ups ("how often do we have to change *it*"), the preceding context clause
  for purely anaphoric requests ("is *that* allowed?"), and session entities for anaphors that point to
  an earlier turn ("can they stay *there*?"). A head-count becomes a capacity facet ("room or venue
  capacity for 30 people: ...") because cross-encoders weight query onsets.
* **Context hedging**: carried context can mislead the reranker ("what class can I fly" + "San
  Francisco conference"), so each contextualised sub-query also carries a context-free variant; both
  readings are scored and the better one wins.
* **Over-fragmentation guard**: sub-queries are merged only if they are near-identical in meaning *and*
  share their own words (so "password length" and "password rotation" stay two needs); fan-out is
  capped at four.

### [3] Retrieval, fusion and reranking

For each sub-query (batched per turn): BM25 over contextualised chunks (`title. section. body`) and bge-
small dense search → **reciprocal-rank fusion** (k=60, variants max-pooled) → the top 10 fused
candidates are scored by the MiniLM-L6 **cross-encoder**. Chunks from documents marked *superseded*
(e.g. the 2024 travel policy) are demoted when their replacement is in the corpus. Inside the same job the
sentence units of the top three chunks are scored by the cross-encoder for answer selection, so this work
also happens during speech. Across sub-queries, evidence is fused again with RRF and de-duplicated
(a chunk serving several needs is kept once and ranked higher).

**Speculation and reuse.** At end-of-speech the final sub-queries are matched to speculative ones by
term set or embedding similarity (≥0.93), but a speculative result is reused **only if it already
contained every entity and number of the final query** - a late "...it's in Bengaluru" invalidates
Bengaluru-less searches. Only unmatched sub-queries are fetched after the utterance.

### [4] Session-aware synthesis

* **Extractive, cited claims.** For each intent the composer selects 1-4 answer units (sentences or
  bullets) by cross-encoder score, bi-encoder similarity, lexical overlap and specificity (figures,
  limits, durations), adds the entity or section heading a bare bullet needs to stand alone, and cites
  the chunk. For enumerations (several venues) one or two units per sibling document are selected; with
  a head-count, rooms too small for the group are skipped.
* **Explicit uncertainty.** An intent is flagged instead of answered when the evidence is extremely weak,
  when a quantity question ("how much is the fee") has no figure of the requested type in the evidence,
  or when the named entity's own document never states an attribute that sibling entities do state
  ("parking at the Sector 62 centre"). Across intents an **entity x aspect** check verifies each entity
  surfaced by the anchor intent against every aspect intent and reports gaps ("catering options for
  Hinjewadi Tech Park Training Suites could not be verified from the corpus").
* **Versioned session memory.** The answer state (intents, claims with citations, uncertainty) is
  versioned. A refinement builds *delta* sub-queries: a new head-count or an explicit replacement
  ("it's Germany now", "not a birth") *modifies* the matching intent (old claims retired, entity
  substituted); any other detail *adds* a delta intent asked in the context of the closest existing
  intent. Earlier claims are retained ("Still applies ...") and the diff (added / retained / retired) is
  emitted. Low-content fragments ("full disclosure", "scratch that") fold into the substantive constraint.
* **Presentation transforms** ("two bullets", "shorter", "repeat") re-render the current version with its
  existing citations: no retrieval and no new claims.
* **Grounding verification.** Every claim is re-checked: each citation must resolve to a real chunk that
  was retrieved for this answer, every number in the claim must appear in the cited text, and most of its
  content words must too. Fabricated IDs are impossible in the extractive path and are counted anyway.
* **Optional LLM phrasing** (`DUPLEXRAG_SYNTHESIS=llm`): any OpenAI-compatible endpoint rewrites only the
  selected evidence; each output sentence is verified and dropped if unsupported, falling back to the
  extractive answer on failure.

### [5] Telemetry

Every stage emits JSON Lines events on both the stream clock and the wall clock: chunk arrival, every
controller decision with the gate label and reason, sub-queries with trigger type, retrieval start and
completion with compute time, reuse or waste, top hits, fused evidence, first token (TTFT), answer version
with diff and citations, grounding result, uncertainty flags and per-turn latency and cost (model tokens,
CPU-ms, USD). The schema is in `schemas/telemetry.schema.json`, and `scripts/check_trace.py` validates traces.

## 3. Data and provenance

The corpus referenced in the brief was not distributed with our theme guide, so we authored a synthetic
knowledge base for a fictional company (46 documents, 203 sections, 1,128 answer units) that mirrors the
guide's scenarios and contains engineered hard cases: missing attributes (two venues without catering
information), a superseded policy, a directory that repeats facts, and near-duplicate topics. Each chunk
keeps `doc_id`, section number, titles, status (`current` / `superseded`, `superseded_by`) and source file,
and every answer claim cites `Doc_ID §N`. The engine contains no corpus-specific strings: salience and
proper-noun vocabularies are derived from the index at build time, and prompts, queries and answers are not
embedded in code. Benchmark splits were written by independent agents that saw only the corpus: dev (18
sessions) for tuning, dev-b (32) used once as held-out and then for diagnosis, and test (30, written after
the engine was frozen) for the final numbers.

## 4. Trade-offs

| Decision | Benefit | Cost / risk | Mitigation |
|---|---|---|---|
| Extractive answers (no LLM by default) | grounded by construction, ~0 cost, ms-level composition | less fluent; multi-sentence answers | section context added to bullets; optional verified LLM mode |
| Speculative retrieval during speech | TTFT hidden behind speech | wasted CPU when the utterance changes | stability gate, entity-aware reuse, telemetry of waste |
| Cross-encoder rerank (22M) | large precision gain over fused lists | ~15 ms per pair on CPU | batching, 10 candidates, run during speech |
| Context hedging | robust to misleading context | ~50% more rerank pairs | only top-5 candidates scored twice |
| Conservative "not found" | avoids suppressing correct answers | some unanswerables get a related, cited answer | typed-quantity, entity-attribute and sibling-coverage checks |
| Rule + small classifier gate | transparent, <5 ms, no training data needed | cue coverage limits | hybrid mode; model trained on domain-general templates |

## 5. Failure modes and mitigations

| Failure mode (brief section 6) | Mitigation in DuplexRAG |
|---|---|
| Premature retrieval on noise | clause stability gate, dangling-word check, re-fire only on new entities/terms, follow-up word/term guard |
| Context loss on late constraints | versioned answer state, delta sub-queries, retained/retired claims, session entity carry-over |
| Citation hallucination | extractive claims, verifier (existence, retrieval membership, numbers, overlap), fabricated-ID counter |
| Querying on presentation-only turns | turn gate + suppression; presentation transforms reuse existing citations |
| Over-fragmented sub-queries | fragment merging (meaning + own-word overlap), low-content fragment folding, fan-out cap |
| Late entity invalidates speculation | reuse only if speculative query had every final entity/number |
| Superseded policy retrieved | status-aware demotion when the replacement exists; current documents preferred in synthesis |
| Missing evidence answered silently | explicit uncertainty flags (typed quantities, entity attributes, entity x aspect coverage) |

## 6. Cost model

Cost per turn = measured CPU time (embedding + reranking + controller + synthesis) x on-demand price
of a vCPU-hour (USD 0.0447, c7i class) x worker threads, plus LLM tokens x price in LLM mode. Model
tokens processed per turn are reported separately.
