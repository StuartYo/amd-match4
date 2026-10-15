"""The retrieval agent.

The LLM plans the search and reads page excerpts; deterministic code fetches,
extracts, checks that every answer really appears on a fetched page, and
decides the confidence. A value the model "remembers" but no page contains is
never returned.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from . import config, extract, sources, web
from .extract import Candidate
from .llm import Usage, get_llm, parse_json

STOP = set("""a an the of to in on for and or is are was were be by with at from as what which who whom whose
when where how does do did can could should would will it its this that these those me my i you your
please find tell give number current latest official""".split())

TYPE_KEYWORDS = [
    ("email", ("email", "e-mail")),
    ("phone", ("phone", "telephone", "call ", "hotline", "toll-free", "toll free", "fax")),
    ("version", ("version", "release", "lts")),
    ("price", ("price", "cost", "how much", "fee", "subscription")),
    ("year", ("what year", "which year", "year was", "year did", "founded in", "established in")),
    ("date", ("what date", "which date", "when was", "when did", "when is", "release date", "date of")),
    ("url", ("website", "url", "homepage", "web site", "link to")),
    ("address", ("address", "headquarters located", "located at")),
    ("number", ("how many", "population", "number of", "height", "length", "capacity")),
]


@dataclass
class Result:
    answer: str
    confidence: float
    sources: list[str]
    answer_type: str
    trace: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    def to_output(self) -> dict:
        return {
            "answer": self.answer,
            "text": self.answer,
            "sources": self.sources,
            "citations": self.sources,
            "confidence": round(self.confidence, 3),
        }


def tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9+#.\-]*", s.lower()) if t not in STOP and len(t) > 1]


def rule_type(q: str) -> str:
    low = " " + q.lower() + " "
    for t, kws in TYPE_KEYWORDS:
        if any(k in low for k in kws):
            return t
    return "text"


# ---------------------------------------------------------------- planning

PLAN_PROMPT = """You plan web searches for a question-answering agent.
Question: {q}

Return ONLY a JSON object with these keys:
"answer_type": one of phone, email, version, date, year, number, price, url, address, name, text
"entity": the organisation, product or thing the question is about
"attribute": what is asked about it, in a few words
"is_latest": true if the question asks for the latest/current value
"ecosystem": pypi, npm, github, or none (only for software packages)
"package": the package or repository name if it is software, else ""
"queries": 2 short web search queries that would find the answer on an official page"""


def plan_question(q: str, usage: Usage) -> dict:
    plan = {"question": q, "answer_type": rule_type(q), "entity": "", "attribute": "",
            "is_latest": any(w in q.lower() for w in ("latest", "current", "newest", "most recent")),
            "ecosystem": "none", "package": "", "queries": [q]}
    try:
        reply = get_llm().chat([{"role": "user", "content": PLAN_PROMPT.format(q=q)}], max_tokens=200, usage=usage)
        data = parse_json(reply) or {}
    except Exception as e:  # model failure must not kill the run
        data = {"_error": type(e).__name__}
    if data.get("answer_type") in extract.ANSWER_TYPES:
        # Rules win for the typed cases they recognise; the model fills the rest.
        if plan["answer_type"] == "text":
            plan["answer_type"] = data["answer_type"]
    for k in ("entity", "attribute", "ecosystem", "package"):
        if isinstance(data.get(k), str):
            plan[k] = data[k].strip()
    if isinstance(data.get("is_latest"), bool):
        plan["is_latest"] = plan["is_latest"] or data["is_latest"]
    qs = [x for x in data.get("queries", []) if isinstance(x, str) and x.strip()] if isinstance(data.get("queries"), list) else []
    plan["queries"] = ([q] + qs)[:3]
    return plan


# ---------------------------------------------------------------- reading

def chunk_page(page: web.Page, size: int = 700) -> list[str]:
    lines, chunks, cur = page.text.split("\n"), [], ""
    for ln in lines:
        if len(cur) + len(ln) > size and cur:
            chunks.append(cur)
            cur = ""
        cur += ln + "\n"
    if cur:
        chunks.append(cur)
    return chunks


def rank_chunks(pages: list[web.Page], plan: dict, limit_chars: int) -> list[tuple[int, str]]:
    qtok = tokens(plan["question"] + " " + plan.get("attribute", "") + " " + plan.get("entity", ""))
    if not qtok:
        return []
    all_chunks = [(i, c) for i, p in enumerate(pages) for c in chunk_page(p)]
    df = Counter()
    toks = []
    for _, c in all_chunks:
        ts = set(tokens(c))
        toks.append(ts)
        df.update(ts)
    n = max(1, len(all_chunks))
    scored = []
    for (i, c), ts in zip(all_chunks, toks):
        s = sum(math.log(1 + n / (1 + df[t])) for t in set(qtok) if t in ts)
        if plan["answer_type"] in ("phone", "email", "version", "price", "year", "date", "number"):
            if extract.find_typed(c, plan["answer_type"]):
                s *= 1.4
        scored.append((s, i, c))
    scored.sort(key=lambda x: -x[0])
    out, used, per_page = [], 0, Counter()
    for s, i, c in scored:
        if s <= 0 or per_page[i] >= 3:
            continue
        if used + len(c) > limit_chars:
            break
        out.append((i, c))
        used += len(c)
        per_page[i] += 1
    return out


READ_PROMPT = """Answer the question using ONLY the numbered web excerpts below.
Question: {q}
Expected answer type: {t}

{ctx}

Rules: give only the value itself (no sentence, no units unless part of the value, no label).
If several values appear, choose the one that best matches every word of the question
(official source, the right country/department, the current/latest one).
If the excerpts do not contain the answer, use an empty answer.
Return ONLY JSON: {{"answer": "...", "source": <excerpt number>}}"""


def llm_read(pages: list[web.Page], plan: dict, usage: Usage, max_time: float | None = None) -> Candidate | None:
    picked = rank_chunks(pages, plan, config.LLM_MAX_CONTEXT_CHARS)
    if not picked:
        return None
    ctx = "\n\n".join(f"[{k + 1}] ({web.domain_of(pages[i].url)})\n{c.strip()}" for k, (i, c) in enumerate(picked))
    reply = get_llm().chat([{"role": "user", "content": READ_PROMPT.format(q=plan["question"], t=plan["answer_type"], ctx=ctx)}],
                           max_tokens=80, usage=usage, max_time=max_time)
    data = parse_json(reply) or {}
    ans = str(data.get("answer") or "").strip()
    if not ans:
        return None
    try:
        k = int(data.get("source")) - 1
        page_i = picked[k][0] if 0 <= k < len(picked) else picked[0][0]
    except (TypeError, ValueError):
        page_i = picked[0][0]
    return Candidate(ans, pages[page_i].url, "llm", context=reply[:200])


# ---------------------------------------------------------------- scoring

def entity_domain_match(url: str, entity: str, official: str = "") -> bool:
    reg = web.registrable(web.domain_of(url))
    if official and reg == official:
        return True
    dom = reg.split(".")[0]
    ents = [t for t in tokens(entity) if len(t) > 2]
    return bool(dom) and any(t.replace("-", "") in dom or dom in t.replace("-", "") for t in ents)


def _vkey(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4])


def score_candidates(cands: list[Candidate], pages: list[web.Page], plan: dict) -> list[Candidate]:
    t = plan["answer_type"]
    latest_version = t == "version" and plan.get("is_latest")
    official = plan.get("official", "")
    qtok = set(tokens(plan["question"] + " " + plan.get("attribute", "")))
    merged: dict[str, Candidate] = {}
    counts: Counter = Counter()
    llm_keys: set = set()
    for c in cands:
        val = extract.format_answer(c.value, t)
        if not val or len(val) > 200:
            continue
        key = extract.grade_norm(val) if t != "phone" else extract.phone_digits(val)[-10:]
        base = {"api": 6.0 if latest_version else 3.0, "api-guess": 2.5, "llm": 2.2, "tel": 1.6, "jsonld": 1.6, "regex": 0.8}.get(c.method, 0.5)
        ctx_overlap = len(qtok & set(tokens(c.context))) if c.method in ("regex", "tel") else 0
        s = base + 0.35 * min(ctx_overlap, 4)
        if t == "phone" and re.search(r"\bfax\b|\btty\b|\btdd\b", c.context.lower()) and "fax" not in plan["question"].lower():
            s -= 1.5
        if t == "version" and c.method == "regex":
            ctx = c.context.lower()
            ents = [x for x in tokens(plan.get("entity", "")) if len(x) > 1]
            if not (re.search(r"version|release|\blts\b|stable|\bv\d", ctx) and (not ents or any(e in ctx for e in ents))):
                continue  # a bare decimal (price, score) is not a version
        if t in ("number", "year", "date", "price") and c.method == "regex" and ctx_overlap == 0:
            continue  # a number with none of the question's words nearby is noise
        if entity_domain_match(c.source, plan.get("entity", ""), official):
            s += 1.5 if official and web.registrable(web.domain_of(c.source)) == official else 1.0
        counts[key] += 1
        if c.method == "llm":
            llm_keys.add(key)
        m = merged.get(key)
        if m is None or s > m.score:
            merged[key] = Candidate(val, c.source, c.method, c.context, s)
    for key, c in merged.items():
        # Repetition helps a little, but a value printed 40 times on one page is still one source.
        c.score += 0.5 * math.log2(counts[key])
        if key in llm_keys and c.method != "llm":
            c.score += 1.5  # the reader picked a value the extractors also found
            c.method += "+llm"
    out = []
    for key, c in merged.items():
        grounded_domains = set()
        for p in pages:
            if p.ok and extract.value_in_text(c.value, p.text + " " + " ".join(p.tel), t):
                grounded_domains.add(web.registrable(web.domain_of(p.url)))
        if not c.method.startswith("api") and not grounded_domains:
            continue  # not on any page we fetched: never answer it
        c.support = grounded_domains
        c.score += 0.6 * min(len(grounded_domains), 4)
        if official and official in grounded_domains:
            c.score += 1.0
        out.append(c)
    if latest_version and out:
        # "1.37" vs "1.37.1": the longer release of the same line wins.
        for a in out:
            for b in out:
                if b is not a and b.value.startswith(a.value + ".") and b.score < a.score:
                    b.score, a.score = a.score + 0.1, a.score - 0.5
        # Among the candidates the question is about, the newest one gets a nudge.
        top_vals = sorted((c for c in out if _vkey(c.value)), key=lambda c: _vkey(c.value), reverse=True)
        if top_vals:
            top_vals[0].score += 0.8
    out.sort(key=lambda c: -c.score)
    return out


def confidence(ranked: list[Candidate]) -> float:
    if not ranked:
        return 0.0
    top = ranked[0].score
    second = ranked[1].score if len(ranked) > 1 else 0.0
    margin = top - second
    c = 1 / (1 + math.exp(-(0.9 * margin + 0.35 * top - 2.0)))
    if len(ranked[0].support) <= 1 and not ranked[0].method.startswith("api"):
        c = min(c, 0.75)  # one page only: cap
    return max(0.05, min(c, 0.97))


# ---------------------------------------------------------------- main loop

def answer(question: str, budget_s: float | None = None) -> Result:
    t0 = time.time()
    budget = budget_s or config.TIME_BUDGET_S
    left = lambda: budget - (time.time() - t0)  # noqa: E731
    usage = Usage()
    trace: dict = {"steps": []}

    plan = plan_question(question, usage)
    trace["plan"] = {k: v for k, v in plan.items() if k != "question"}
    t = plan["answer_type"]

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    wd_job = pool.submit(web.wikidata_entity, plan.get("entity", "")) if plan.get("entity") else None
    hits = web.search(plan["queries"][:2], timeout_s=min(8.0, left() - 8)) if left() > 10 else []
    try:
        wd = wd_job.result(timeout=max(0.2, min(3.0, left() - 8))) if wd_job else None
    except Exception:
        wd = None
    pool.shutdown(wait=False, cancel_futures=True)
    homepage = (wd or {}).get("website", "")
    if not homepage:
        # No Wikidata answer: take the first search hit on a domain named after the entity.
        for h in hits:
            if entity_domain_match(h.url, plan.get("entity", "")) and "wikipedia" not in h.url:
                homepage = f"https://www.{web.registrable(web.domain_of(h.url))}/"
                break
    plan["official"] = web.registrable(web.domain_of(homepage)) if homepage else ""
    trace["wikidata"] = wd
    trace["steps"].append(f"search {len(hits)} hits @{time.time() - t0:.1f}s")
    gh_repos = []
    for h in hits:
        m = re.match(r"https?://github\.com/([\w.-]+/[\w.-]+)", h.url)
        if m and m.group(1) not in gh_repos and not m.group(1).endswith(".github.io"):
            gh_repos.append(m.group(1))

    cands: list[Candidate] = []
    if t == "version" and plan.get("is_latest"):
        cands += sources.version_candidates(plan, gh_repos)

    # Prefer the entity's own domain, then the fused search order.
    urls = [h.url for h in hits]
    urls.sort(key=lambda u: 0 if entity_domain_match(u, plan.get("entity", ""), plan["official"]) else 1)
    urls = urls[: config.MAX_PAGES]
    if homepage and t in ("phone", "email", "address", "url") and not any(
            u.rstrip("/") == homepage.rstrip("/") for u in urls):
        urls = [homepage] + urls[: config.MAX_PAGES - 1]
    pages = web.fetch_many(urls, timeout_s=min(7.0, left() - 7)) if urls and left() > 9 else []
    # Search snippets are a cheap extra "page": they often carry the value directly.
    snippet_page = web.Page(url="search://snippets", ok=True,
                            text="\n".join(f"{h.title}. {h.snippet}" for h in hits[:12]))
    trace["steps"].append(f"fetched {sum(p.ok for p in pages)}/{len(pages)} @{time.time() - t0:.1f}s")

    if left() > 12:
        follow = []
        for p in pages:
            if p.ok and entity_domain_match(p.url, plan.get("entity", ""), plan["official"]):
                follow += web.follow_candidates(p, t, 2)
        follow = [u for u in dict.fromkeys(follow) if u not in urls][: config.MAX_FOLLOW]
        if follow:
            pages += web.fetch_many(follow, timeout_s=min(5.0, left() - 8))
            trace["steps"].append(f"followed {len(follow)} @{time.time() - t0:.1f}s")

    ok_pages = [p for p in pages if p.ok]
    trace["pages"] = [p.url for p in ok_pages]
    for p in ok_pages:
        for v, ctx in extract.find_typed(p.text, t)[:40]:
            cands.append(Candidate(v, p.url, "regex", ctx))
        if t == "phone":
            cands += [Candidate(v, p.url, "tel", "tel link " + p.title) for v in p.tel]
        if t == "email":
            cands += [Candidate(v, p.url, "tel", "mailto link " + p.title) for v in p.mailto]
        for item in p.jsonld:
            key = {"phone": "telephone", "email": "email", "url": "url"}.get(t)
            if key and isinstance(item.get(key), str):
                cands.append(Candidate(item[key], p.url, "jsonld", f"json-ld {item.get('@type')} {item.get('name')}"))

    read_pages = ok_pages + [snippet_page]
    if left() > 3:
        try:
            c = llm_read(read_pages, plan, usage, max_time=max(1.0, left() - 1.5))
            if c:
                cands.append(c)
                trace["llm_answer"] = c.value
        except Exception as e:
            trace["llm_error"] = type(e).__name__

    ranked = score_candidates(cands, read_pages, plan)
    trace["top"] = [(c.value, round(c.score, 2), c.method, sorted(c.support)[:4]) for c in ranked[:5]]
    conf = confidence(ranked)
    if not ranked or conf < config.ABSTAIN_BELOW:
        res = Result("", conf, [], t, trace, usage)
    else:
        best = ranked[0]
        srcs = [best.source] if not best.source.startswith("search://") else []
        for p in ok_pages:
            if p.url not in srcs and web.registrable(web.domain_of(p.url)) in best.support and len(srcs) < 3:
                if extract.value_in_text(best.value, p.text, t):
                    srcs.append(p.url)
        res = Result(best.value, conf, srcs, t, trace, usage)
    trace["seconds"] = round(time.time() - t0, 2)
    return res


# ---------------------------------------------------------------- baseline

def baseline_answer(question: str) -> Result:
    """Control condition: the same model, no web, no tools."""
    usage = Usage()
    t = rule_type(question)
    reply = get_llm().chat([
        {"role": "user", "content": f"Question: {question}\nAnswer with only the value itself, no sentence. "
                                    "If you are not sure, give your best guess."}], max_tokens=40, usage=usage)
    val = extract.format_answer(reply.splitlines()[0] if reply else "", t)
    return Result(val, 0.5, [], t, {}, usage)
