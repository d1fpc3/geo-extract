"""Text and name normalization.

Business names show up in AI answers in many shapes: "Joe's Plumbing LLC" becomes
"Joes Plumbing", "Eco-Cool" becomes "EcoCool", "&" becomes "and". Everything here maps
those shapes onto one comparable form while keeping track of where each token came from
in the original text, so evidence spans always point at the real characters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse

# Legal suffixes that never distinguish one business from another.
SUFFIXES = frozenset(
    {
        "llc",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "pllc",
        "lp",
        "llp",
        "pc",
        "pa",
    }
)

# Words that describe a trade rather than a business. A name made only of these plus a
# town ("Bel Air Plumbing") has to match in full; it can't match on a fragment.
GENERIC = frozenset(
    {
        "the",
        "and",
        "of",
        "a",
        "an",
        "by",
        # trades
        "hvac",
        "heating",
        "heat",
        "cooling",
        "air",
        "conditioning",
        "ac",
        "furnace",
        "plumbing",
        "plumber",
        "plumbers",
        "drain",
        "drains",
        "rooter",
        "sewer",
        "pipe",
        "electric",
        "electrical",
        "electrician",
        "electricians",
        "power",
        "roofing",
        "roofer",
        "roofers",
        "roof",
        "roofs",
        "gutters",
        "gutter",
        "lawn",
        "lawns",
        "landscape",
        "landscapes",
        "landscaping",
        "turf",
        "yard",
        "tree",
        "pool",
        "pools",
        "spa",
        "med",
        "medspa",
        "aesthetics",
        "skin",
        "dental",
        "dentistry",
        "dentist",
        "smile",
        "smiles",
        "orthodontics",
        "family",
        "auto",
        "detail",
        "detailing",
        "mobile",
        "wash",
        "car",
        "pest",
        "control",
        "termite",
        "exterminating",
        "garage",
        "door",
        "doors",
        "cleaning",
        "repair",
        "repairs",
        "mechanical",
        "contractors",
        "contracting",
        "construction",
        "systems",
        "solutions",
        "services",
        "service",
        "care",
        "group",
        "pros",
        "pro",
        "experts",
        "home",
        "homes",
        "center",
        "studio",
        "clinic",
        "comfort",
        "climate",
        # legal suffixes are generic too
        *SUFFIXES,
    }
)

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|&")
_URL = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
_BARE_DOMAIN = re.compile(
    r"(?<![@\w.-])((?:[a-z0-9-]+\.)+(?:com|net|org|co|us|biz|io|info|pro|services))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Token:
    norm: str
    start: int
    end: int
    capitalized: bool


def fold(text: str) -> str:
    """Lowercase and strip accents ("Lumière" -> "lumiere")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _norm_word(raw: str) -> str:
    if raw == "&":
        return "and"
    word = fold(raw).replace("’", "'")
    # Possessives and contractions collapse: "joe's" -> "joes".
    return word.replace("'", "")


def tokenize(text: str) -> list[Token]:
    """Split text into normalized word tokens with their original character offsets."""
    return [
        Token(
            norm=_norm_word(m.group(0)),
            start=m.start(),
            end=m.end(),
            capitalized=m.group(0)[:1].isupper() or m.group(0)[:1].isdigit(),
        )
        for m in _WORD.finditer(text)
    ]


def name_tokens(name: str) -> list[str]:
    """Normalized name tokens, minus a leading "the" and trailing legal suffixes."""
    toks = [t.norm for t in tokenize(name)]
    while toks and toks[0] == "the":
        toks.pop(0)
    while len(toks) > 1 and toks[-1] in SUFFIXES:
        toks.pop()
    return toks


def distinctive_tokens(name: str, extra_generic: frozenset[str] = frozenset()) -> list[str]:
    """The part of a name that identifies the business ("Polar Bear" in "Polar Bear HVAC").

    `extra_generic` is used for location words from the query: in an answer about Bel Air,
    the words "bel" and "air" can't identify "Bel Air Plumbing" on their own.
    """
    generic = GENERIC | extra_generic
    return [t for t in name_tokens(name) if t not in generic]


def registered_domain(url_or_domain: str) -> str:
    """`https://www.Example.com/path` -> `example.com`."""
    raw = url_or_domain.strip()
    host = urlparse(raw).netloc if "://" in raw else raw.split("/")[0]
    host = host.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    labels = [label for label in host.split(".") if label]
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def find_domains(text: str) -> list[tuple[str, int, int]]:
    """Every URL or bare domain in the text as (registered_domain, start, end)."""
    found: list[tuple[str, int, int]] = []
    taken: list[tuple[int, int]] = []
    for m in _URL.finditer(text):
        found.append((registered_domain(m.group(0)), m.start(), m.end()))
        taken.append((m.start(), m.end()))
    for m in _BARE_DOMAIN.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue
        found.append((registered_domain(m.group(1)), m.start(1), m.end(1)))
    return sorted(found, key=lambda d: d[1])


def location_words(location: str | None) -> frozenset[str]:
    """Words of the query's location ("Bel Air, MD" -> {"bel", "air", "md"})."""
    if not location:
        return frozenset()
    return frozenset(t.norm for t in tokenize(location))
