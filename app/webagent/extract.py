"""Typed candidate extraction, grading normalization and answer formatting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ANSWER_TYPES = ("phone", "email", "version", "date", "year", "number", "price", "url", "address", "name", "text")


@dataclass
class Candidate:
    value: str
    source: str  # url
    method: str  # regex | llm | api | jsonld | tel
    context: str = ""
    score: float = 0.0
    support: set = field(default_factory=set)  # registrable domains that contain the value


def grade_norm(s: str) -> str:
    """Same normalization the challenge grader used in Matches 2 and 3."""
    s = s.upper()
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[-.·_]", "", s)


# ---------------------------------------------------------------- phone

_PHONE = re.compile(
    r"(?<![\w/])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{1,4}\)[\s.\-]?)?\d{2,4}[\s.\-]\d{2,4}(?:[\s.\-]\d{2,5}){0,2}(?![\w/])"
)
_VANITY = re.compile(r"\b1[\s\-.]?8(?:00|33|44|55|66|77|88)[\s\-.][A-Z0-9]{3}[\s\-.][A-Z0-9]{4}\b")


def phone_digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def looks_like_phone(s: str) -> bool:
    d = phone_digits(s)
    if not 7 <= len(d) <= 15:
        return False
    if re.fullmatch(r"(19|20)\d{2}[\s\-.]?(19|20)\d{2}", s.strip()):
        return False  # year range
    if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", s.strip()):
        return False  # date
    return True


def find_phones(text: str) -> list[tuple[str, str]]:
    out = []
    for m in list(_PHONE.finditer(text)) + list(_VANITY.finditer(text)):
        val = m.group(0).strip(" .-")
        if looks_like_phone(val):
            ctx = text[max(0, m.start() - 90): m.end() + 40].replace("\n", " ")
            out.append((val, ctx))
    return out


def format_phone(s: str) -> str:
    raw = s.strip()
    d = phone_digits(raw)
    if re.search(r"[A-Za-z]", raw):  # vanity number such as 1-800-FLY-DELTA
        return re.sub(r"\s+", "-", raw.replace("(", "").replace(")", ""))
    nanp = (len(d) == 10 and not raw.startswith("+")) or (
        len(d) == 11 and d.startswith("1") and (raw.startswith("+1") or raw.startswith("1"))
    )
    if nanp:
        d = d[-10:]
        return f"{d[0:3]}-{d[3:6]}-{d[6:]}"
    # International: keep the leading +, drop parentheses (the grader keeps them).
    s = s.replace("(", "").replace(")", "")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- others

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_VERSION = re.compile(r"(?<![\w.])v?(\d+\.\d+(?:\.\d+){0,2}(?:[-.]?(?:rc|beta|alpha|b|a|post|dev)\d*)?)(?![\w.]*\d)")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_PRICE = re.compile(r"(?:US\$|\$|USD\s?|EUR\s?|€|£|GBP\s?|¥|JPY\s?|NT\$)\s?\d[\d,]*(?:\.\d{1,2})?")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.]*\d)")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
_DATE = re.compile(
    rf"\b(?:(?:{_MONTHS})\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}|\d{{1,2}} (?:{_MONTHS})\.? \d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b"
)


def find_typed(text: str, answer_type: str) -> list[tuple[str, str]]:
    if answer_type == "phone":
        return find_phones(text)
    pat = {
        "email": _EMAIL, "version": _VERSION, "year": _YEAR, "price": _PRICE,
        "date": _DATE, "number": _NUMBER,
    }.get(answer_type)
    if pat is None:
        return []
    out = []
    for m in pat.finditer(text):
        val = m.group(1) if (pat is _VERSION) else m.group(0)
        if answer_type == "email" and val.lower().endswith((".png", ".jpg", ".gif", ".webp", ".svg")):
            continue
        ctx = text[max(0, m.start() - 90): m.end() + 40].replace("\n", " ")
        out.append((val, ctx))
    return out


def format_answer(value: str, answer_type: str) -> str:
    v = value.strip().strip('"\'').strip()
    v = re.sub(r"\s+", " ", v)
    if answer_type == "phone":
        # "1-888-INFO-FDA (1-888-463-6332)": prefer the all-digit number.
        nums = [p for p, _ in find_phones(v) if not re.search(r"[A-Za-z]", p)]
        if nums:
            return format_phone(nums[0])
        if looks_like_phone(v):
            return format_phone(v)
    if answer_type == "version":
        m = _VERSION.search(v)
        return m.group(1) if m else v.lstrip("vV")
    if answer_type == "email":
        m = _EMAIL.search(v)
        return m.group(0).lower() if m else v
    if answer_type == "year":
        m = _YEAR.search(v)
        return m.group(0) if m else v
    if answer_type in ("url",):
        return v.rstrip("/")
    return v.rstrip(".")


def value_in_text(value: str, text: str, answer_type: str) -> bool:
    """Grounding check: does the page really contain this value?"""
    if not value:
        return False
    if answer_type == "phone":
        d = phone_digits(value).removeprefix("1") if len(phone_digits(value)) == 11 else phone_digits(value)
        for val, _ in find_phones(text):
            pd = phone_digits(val)
            if len(pd) == 11 and pd.startswith("1"):
                pd = pd[1:]
            if pd == d:
                return True
        return False
    return grade_norm(value) in grade_norm(text)
