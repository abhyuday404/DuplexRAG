from duplexrag.decompose import normalize_spoken, spoken_count
from duplexrag.grounding import extract_citations, support_score
from duplexrag.text import content_tokens, numbers_in, split_sentences, stem


def test_spoken_numbers():
    assert normalize_spoken("around a hundred and ten people") == "around 110 people"
    assert normalize_spoken("error eight oh nine") == "error 809"
    assert normalize_spoken("lands at eleven thirty at night") == "lands at 11 30 at night"
    assert normalize_spoken("a fifteen hour flight") == "a 15 hour flight"


def test_self_repairs():
    assert normalize_spoken("what hotel, sorry, what flight class") == "what flight class"
    assert normalize_spoken("hotel cap in Mumbai, no sorry, Hyderabad") == "hotel cap in Hyderabad"
    assert normalize_spoken("the limit in Bengaluru, no sorry, I meant Chennai") == "the limit in Chennai"


def test_aliases_and_fillers():
    t = normalize_spoken("so um it was in pounds and I'm flying to the US, you know")
    assert "GBP" in t and "USA" in t and " um " not in f" {t} " and "you know" not in t


def test_stemming_is_consistent():
    assert stem("images") == stem("image")
    assert stem("charged") == stem("charge") == stem("charges")
    assert stem("taxes") == "tax"


def test_numbers_and_tokens():
    assert numbers_in("INR 2,00,000 and 45 people") == {"200000", "45"}
    assert "200000" in content_tokens("INR 2,00,000")


def test_sentence_units():
    units = split_sentences("Intro sentence here now.\n\n- first bullet item text\n- second bullet item text")
    assert units == ["Intro sentence here now.", "first bullet item text", "second bullet item text"]


def test_spoken_count():
    assert spoken_count("repeat that in two bullets") == 2
    assert spoken_count("put that in 3 quick bullet points") == 3


def test_grounding_helpers():
    assert extract_citations("A claim. [Doc_03 §2] Another [Doc_05 §3, Doc_05 §4]") == \
        ["Doc_03 §2", "Doc_05 §3", "Doc_05 §4"]
    assert support_score("Suite B holds 32 people.", ["Suite B: 32 classroom people"]) > 0.6
    assert support_score("Suite B holds 99 people.", ["Suite B: 32 classroom people"]) == 0.0
