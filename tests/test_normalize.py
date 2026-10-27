from geo_extract.normalize import (
    distinctive_tokens,
    find_domains,
    fold,
    location_words,
    name_tokens,
    registered_domain,
    tokenize,
)


def test_fold_strips_accents_and_case() -> None:
    assert fold("Lumière") == "lumiere"


def test_possessive_and_ampersand() -> None:
    assert [t.norm for t in tokenize("Joe's Pool & Spa")] == ["joes", "pool", "and", "spa"]


def test_tokens_keep_original_offsets() -> None:
    text = "Call Eco-Cool today"
    toks = tokenize(text)
    assert [text[t.start : t.end] for t in toks] == ["Call", "Eco", "Cool", "today"]


def test_name_tokens_drop_legal_suffixes_and_leading_the() -> None:
    assert name_tokens("The Ironclad Roofing Co.") == ["ironclad", "roofing"]
    assert name_tokens("ClearFlow Plumbing LLC") == ["clearflow", "plumbing"]


def test_single_word_suffix_name_is_kept() -> None:
    assert name_tokens("Company") == ["company"]


def test_distinctive_tokens_remove_trade_and_location_words() -> None:
    assert distinctive_tokens("Polar Bear Heating & Air") == ["polar", "bear"]
    loc = location_words("Bel Air, MD")
    assert distinctive_tokens("Bel Air Plumbing", loc) == []


def test_registered_domain() -> None:
    assert registered_domain("https://www.PolarBearAir.com/services?x=1") == "polarbearair.com"
    assert registered_domain("glossbossdetail.com") == "glossbossdetail.com"


def test_find_domains_urls_and_bare() -> None:
    text = "See https://www.a-one.com/x and also b2b.net for more."
    assert [d for d, _, _ in find_domains(text)] == ["a-one.com", "b2b.net"]


def test_find_domains_ignores_emails() -> None:
    assert find_domains("email me at joe@joespools.com") == []
