# Telemetry and observability schema

DuplexRAG writes one JSON object per event (JSON Lines). The formal schema is
[`schemas/telemetry.schema.json`](../schemas/telemetry.schema.json); `python scripts/check_trace.py
<trace.jsonl>` validates a trace against it and reports **G6 trace coverage** (share of turns whose
trace contains every required event, with timestamps, version and cost fields).

Where traces come from:

* `duplexrag replay FILE.jsonl` → `runs/replay/trace.jsonl`
* `duplexrag bench ...` → `results/<split>/<config>/trace.jsonl`
* the web demo streams the same events over the WebSocket; **Trace ⤓** downloads the session trace.

## Clocks

| field | meaning |
|---|---|
| `t_stream` | seconds on the stream clock, the same clock as transcript chunk timestamps (session start = 0). In replays it is the discrete-event clock, and retrieval jobs sit at their dispatch time plus their measured CPU time. |
| `t_wall` | Unix epoch seconds when the event was emitted |
| `seq` | monotonic sequence number per session |

## Events

| event | emitted | key fields |
|---|---|---|
| `session_started` | once | `config` (controller, retrieval mode, models, corpus size) |
| `turn_started` | per turn | - |
| `chunk_received` | per ASR chunk | `text`, `partial` (utterance so far) |
| `controller_decision` | per chunk + once at end of speech | `action` (wait / retrieve / suppress / finalize), `gate` {label, confidence, reason, source}, `reason`, `queries`, `final` |
| `decomposition` | end of speech (retrieval / refinement) | `clauses` (text, kind, salient terms), `sub_queries`, `scope` (`delta` for refinements), `targets` (modify / add) |
| `retrieval_started` | per sub-query dispatch | `qid`, `query`, `keywords`, `trigger` (provisional / multi_intent / refinement / final), `origin` clause |
| `retrieval_completed` | per retrieval job | `qids`, `t_dispatch`, `t_start`, `compute_ms`, `used` (false = speculative result superseded), `top` hits, `stats` (cache hits, pairs reranked) |
| `evidence_fused` | end of speech | fused chunk labels, per-query evidence |
| `retrieval_suppressed` | presentation / chit-chat turns | `reason`, `detail` |
| `answer_started`, `first_token` | answer turns | `ttft_ms` (end of speech → first token) |
| `answer_version` | every turn | `version`, `kind`, `text`, `citations`, `diff` {parent_version, mode: new_answer / refine / presentation, added, retained, retired}, `view` |
| `grounding_check` | every turn | `claims`, `supported`, `support_rate`, `fabricated_ids`, `unretrieved_ids` |
| `uncertainty_flag` | per flag | `intent_id`, `doc_id` (entity x aspect note), `message` |
| `turn_completed` | every turn | `kind`, `latency` {ttft_ms, first_retrieval_s, retrieval_lead_ms, post_utterance_retrieval_ms, synthesis_ms, controller_ms_total}, `cost` {embed_tokens, rerank_tokens, cpu_ms, usd, llm_tokens_in/out}, `speculative` {dispatched, reused, wasted}, `early_retrieval` |
| `session_ended` | once | final `versions` |

Required per turn (G6): `turn_started`, `chunk_received`, `controller_decision`, `answer_version`,
`grounding_check`, `turn_completed`, plus `retrieval_started` and `retrieval_completed` whenever the turn retrieved.

## Example (guide Example 1, abridged)

```json
{"event":"chunk_received","turn_id":"t1","t_stream":2.09,"text":"So um I need","partial":"So um I need"}
{"event":"controller_decision","turn_id":"t1","t_stream":2.09,"action":"wait","gate":{"label":"undecided","reason":"waiting for content"}}
{"event":"retrieval_started","turn_id":"t1","t_stream":4.87,"qid":"t1p0_1","query":"customer workshop in Pune","trigger":"provisional"}
{"event":"retrieval_started","turn_id":"t1","t_stream":6.65,"query":"room or venue capacity for 30 people: customer workshop in Pune","trigger":"provisional"}
{"event":"retrieval_started","turn_id":"t1","t_stream":10.04,"query":"cancellation policy for an event with 30 people Pune","trigger":"multi_intent"}
{"event":"retrieval_started","turn_id":"t1","t_stream":11.72,"query":"catering options Pune","trigger":"multi_intent"}
{"event":"controller_decision","turn_id":"t1","t_stream":12.22,"action":"finalize","gate":{"label":"retrieval"}}
{"event":"first_token","turn_id":"t1","t_stream":12.28,"ttft_ms":58.7}
{"event":"answer_version","turn_id":"t1","version":1,"kind":"retrieval","citations":["Doc_05 §2","Doc_03 §2","Doc_04 §2","Doc_05 §3","..."],"diff":{"mode":"new_answer"}}
{"event":"grounding_check","turn_id":"t1","claims":12,"supported":12,"support_rate":1.0,"fabricated_ids":[]}
{"event":"uncertainty_flag","turn_id":"t1","doc_id":"Doc_04","message":"Catering options for Hinjewadi Tech Park Training Suites, Pune could not be verified from the corpus."}
{"event":"turn_completed","turn_id":"t1","latency":{"ttft_ms":58.67,"retrieval_lead_ms":7356.0},"cost":{"rerank_tokens":11553,"cpu_ms":3142.8,"usd":0.000156},"speculative":{"dispatched":4,"reused":3,"wasted":1}}
```

## Cost model

`usd = cpu_ms / 3.6e6 x 0.0447 (USD per vCPU-hour, c7i on-demand) x threads + LLM tokens x price`.
CPU time counts the embedding and cross-encoder calls, the controller and synthesis, including wasted
speculative work. In the default extractive mode there are no LLM tokens.
