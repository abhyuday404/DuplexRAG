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
