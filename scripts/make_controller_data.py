"""Generate the training set for the model-based turn gate (data/controller/train.jsonl).

Templates are deliberately domain-general (travel, events, HR, IT, banking, retail,
healthcare, education, ...) so the classifier learns *conversational function*
(new request / refinement / presentation / chit-chat), not the demo corpus. None of
these utterances come from the benchmark splits.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)
OUT = Path(__file__).resolve().parents[1] / "data" / "controller" / "train.jsonl"

TOPICS = [
    ("hotel booking limit in Mumbai", "hotel cap mumbai"), ("parental leave policy", "parental leave"),
    ("return policy for electronics", "return electronics"), ("wifi password reset steps", "wifi password"),
    ("gym membership reimbursement", "gym reimbursement"), ("conference room booking rules", "room booking"),
    ("visa processing time for Canada", "visa canada"), ("laptop warranty coverage", "laptop warranty"),
    ("overtime pay rules", "overtime pay"), ("expense approval workflow", "expense approval"),
    ("interest rate on the savings account", "interest savings"), ("shipping charges to Chennai", "shipping chennai"),
    ("clinic appointment cancellation fee", "clinic cancellation"), ("exam re-evaluation deadline", "exam deadline"),
    ("catering menu for the offsite", "catering offsite"), ("printer driver installation", "printer driver"),
    ("remote work allowance", "remote allowance"), ("car rental approval", "car rental"),
    ("health insurance coverage for parents", "insurance parents"), ("password expiry period", "password expiry"),
    ("meal allowance in London", "meal allowance london"), ("venue capacity in Hyderabad", "venue hyderabad"),
    ("training budget per year", "training budget"), ("phone screen flickering fix", "screen flicker"),
    ("refund timeline for cancelled flights", "refund flights"), ("library late fee", "library fee"),
]
QUESTION_T = [
    "what's the {t}?", "can you tell me about the {t}", "um so what is the {t} exactly", "how does the {t} work",
    "I need to know the {t}", "quick question about the {t}", "do you know the {t}",
    "so like what are the rules on {t}", "tell me the {t} please", "what do we have on {t}",
    "hey what's the deal with {t}", "I was wondering about the {t}", "is there anything on {t}",
    "hi, um, I need help with the {t}",
]
COMPOUND_T = ["what's the {t} and also the {u}", "I need the {t}, the {u}, and um whatever else applies",
              "so {t}, and how about {u}?", "tell me about {t} and {u}"]
FOLLOWUP_Q_T = ["and how long does that usually take?", "and who approves it?", "also what about the {u}?",
                "and if it still doesn't work, can I use a different laptop for the {u}",
                "okay and what do visitors need to bring to the {u}", "also when is my {u} due",
                "okay and is there a form for that?", "and where do I submit it?", "and what's the deadline for it?",
                "so how much does it cost?", "and can I do it online?"]
REFINE_T = [
    "oh wait, actually it's for {n} people, not {m}", "actually the trip is to {place} now", "oh and it's in {place}",
    "wait, I forgot, I'm a {role}", "hmm, what if we cancel {n} days before?", "oh one thing, it's international",
    "actually make it {n} nights instead", "by the way the booking was made after the trip", "oh wait it's a customer event",
    "turns out it's going to be {n} people", "also it'll be on a weekend", "oh, I should mention it's a {role} trip",
    "actually no, change it to {place}", "and if it's more than {n} hours?", "wait, the flight is actually {n} hours",
    "does that change if it's international?", "realistically it'd be about {n} weeks out", "oh we're adopting, not birth",
    "the receipt was in {cur} by the way", "it includes breakfast, does that change anything?",
    "actually I got promoted to {role} last month", "I'm travelling with my family too",
    "hmm and what if I'm on probation?", "oh and we'll need wheelchair access", "it's going to be in {month} actually",
    "oh, and full disclosure, it was actually like {n} weeks ago", "oh I forgot to say, we've got a band lined up",
    "oh, should've mentioned, I already paid for it last week", "oh hang on, I should've said why, it got lost",
    "ah wait, I totally forgot to mention, it's in like {n} days, is that gonna be a problem",
    "oh wait, I just looked at the dates again, it's actually {n} nights, not {m}",
    "I just heard a few {role}s might join too, so I guess it's a bigger event now",
    "hold on, it's actually for {n} people", "to be clear, it's a {role} trip", "turns out it's in {place} now",
]
PRES_T = ["can you repeat that in {k} bullets?", "say that again but shorter", "just give me the gist", "shorten that please",
          "put that in {k} bullet points", "can you summarize that", "repeat the last part", "tl;dr please",
          "could you rephrase that more simply?", "give me that as a quick list", "in one line please",
          "sum it up for me", "okay so in short?", "read that back to me", "can you make that briefer",
          "just the key points please", "say it again slowly", "just give me the short version please",
          "okay, just give me the one-line version", "sorry, I missed that, can you repeat the last bit",
          "what was that last part again?", "can you say that again?", "give me the main points"]
CHIT_T = ["okay thanks", "thank you so much", "great, that's helpful", "cool, got it", "perfect thanks",
          "hi there", "hello", "okay", "alright, sounds good", "that's all for now", "thanks, bye", "awesome",
          "makes sense, thank you", "good morning", "hmm okay", "no that's everything", "brilliant, cheers",
          "okay cool thank you for that", "appreciate it", "yeah that works", "okay cool, thanks for checking",
          "great, thanks for looking into it", "perfect, that's all I needed", "got it, cheers"]
PLACES = ["Pune", "Germany", "Singapore", "Chennai", "London", "Hyderabad", "Japan", "Noida", "the US", "Dubai"]
ROLES = ["director", "VP", "manager", "contractor", "intern", "senior engineer"]
CURS = ["euros", "pounds", "dollars", "yen"]
MONTHS = ["November", "December", "March", "July"]


def fill(t: str) -> str:
    (a, _), (b, _) = random.sample(TOPICS, 2)
    return t.format(t=a, u=b, n=random.choice([3, 10, 14, 25, 40, 50, 120]), m=random.choice([20, 30, 60]),
                    place=random.choice(PLACES), role=random.choice(ROLES), cur=random.choice(CURS),
                    month=random.choice(MONTHS), k=random.choice(["two", "three", "2", "3"]))


def main() -> None:
    rows = []
    for t in QUESTION_T:
        for _ in range(3):
            tp = random.choice(TOPICS)
            rows.append({"text": t.format(t=tp[0]), "prev": False, "label": "retrieval"})
            other = random.choice(TOPICS)
            rows.append({"text": t.format(t=tp[0]), "prev": True, "prev_topic": other[1], "label": "retrieval"})
    for t in COMPOUND_T:
        for _ in range(3):
            rows.append({"text": fill(t), "prev": random.random() < 0.5, "prev_topic": random.choice(TOPICS)[1],
                         "label": "retrieval"})
    for t in FOLLOWUP_Q_T:
        for _ in range(2):
            rows.append({"text": fill(t), "prev": True, "prev_topic": random.choice(TOPICS)[1], "label": "retrieval"})
    for t in REFINE_T:
        for _ in range(3):
            rows.append({"text": fill(t), "prev": True, "prev_topic": random.choice(TOPICS)[1], "label": "refinement"})
    for t in PRES_T:
        for _ in range(3):
            rows.append({"text": fill(t), "prev": True, "prev_topic": random.choice(TOPICS)[1],
                         "label": "presentation"})
    for t in CHIT_T:
        for prev in (False, True):
            rows.append({"text": t, "prev": prev, "prev_topic": random.choice(TOPICS)[1], "label": "chitchat"})
    random.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), Counter(r["label"] for r in rows))


if __name__ == "__main__":
    main()
