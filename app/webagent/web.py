"""Keyless web access: search engines, page fetching, HTML to text, link following."""
from __future__ import annotations

import html as htmllib
import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from . import config

# Sites that rarely hold the primary answer and often hold stale copies.
LOW_VALUE_DOMAINS = (
    "pinterest.", "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
    "youtube.com", "quora.com", "linkedin.com", "scribd.com", "slideshare.net",
)


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    engine: str
    rank: int


@dataclass
class Page:
    url: str
    title: str = ""
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)  # (anchor text, absolute url)
    tel: list[str] = field(default_factory=list)
    mailto: list[str] = field(default_factory=list)
    jsonld: list[dict] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    via: str = "fetch"  # fetch | api


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=config.FETCH_TIMEOUT_S,
        follow_redirects=True,
    )


def domain_of(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def registrable(url_or_host: str) -> str:
    host = domain_of(url_or_host) if "/" in url_or_host else url_or_host.lower()
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "edu", "net") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


# ---------------------------------------------------------------- search

def _ddg(query: str, client: httpx.Client) -> list[SearchHit]:
    r = client.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "us-en"})
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    hits = []
    for i, res in enumerate(soup.select("div.result")):
        a = res.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [""])[0])
        if not href.startswith("http") or "duckduckgo.com/y.js" in href:
            continue
        snip = res.select_one(".result__snippet")
        hits.append(SearchHit(href, a.get_text(" ", strip=True), snip.get_text(" ", strip=True) if snip else "", "ddg", i))
    return hits


def _bing(query: str, client: httpx.Client) -> list[SearchHit]:
    r = client.get("https://www.bing.com/search", params={"q": query, "setlang": "en", "cc": "US"})
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    hits = []
    for i, li in enumerate(soup.select("li.b_algo")):
        a = li.select_one("h2 a")
        if not a:
            continue
        href = a.get("href", "")
        if "bing.com/ck/a" in href:
            m = re.search(r"[?&]u=a1([^&]+)", href)
            if m:
                import base64
                try:
                    pad = m.group(1) + "=" * (-len(m.group(1)) % 4)
                    href = base64.urlsafe_b64decode(pad).decode("utf-8", "ignore")
                except Exception:
                    continue
        if not href.startswith("http"):
            continue
        snip = li.select_one(".b_caption p") or li.select_one("p")
        hits.append(SearchHit(href, a.get_text(" ", strip=True), snip.get_text(" ", strip=True) if snip else "", "bing", i))
    return hits


def _brave(query: str, client: httpx.Client) -> list[SearchHit]:
    r = client.get("https://search.brave.com/search", params={"q": query, "source": "web"})
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    hits = []
    for d in soup.select('div.snippet[data-type="web"]'):
        a = d.select_one("a[href^=http]")
        if not a:
            continue
        t = d.select_one(".title")
        c = d.select_one(".snippet-description, .generic-snippet, .content")
        hits.append(SearchHit(a["href"], t.get_text(" ", strip=True) if t else a.get_text(" ", strip=True),
                              c.get_text(" ", strip=True) if c else "", "brave", len(hits)))
    return hits


def _harvest(html_text: str, own: tuple[str, ...], engine: str) -> list[SearchHit]:
    """Engine-agnostic parser: outbound links with a real title, in page order."""
    soup = BeautifulSoup(html_text, "lxml")
    hits, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("http") or any(o in domain_of(href) for o in own):
            continue
        title = a.get_text(" ", strip=True)
        key = href.split("#")[0]
        if len(title) < 12 or key in seen:
            continue
        seen.add(key)
        block = a
        for _ in range(4):
            if block.parent is None or len(block.parent.get_text(" ", strip=True)) > 600:
                break
            block = block.parent
        snippet = block.get_text(" ", strip=True).replace(title, "", 1)[:400]
        hits.append(SearchHit(key, title, snippet, engine, len(hits)))
    return hits


def _seznam(query: str, client: httpx.Client) -> list[SearchHit]:
    r = client.get("https://search.seznam.cz/", params={"q": query})
    if r.status_code != 200:
        return []
    return _harvest(r.text, ("seznam.cz", "szn.cz", "mapy.cz", "firmy.cz", "novinky.cz", "sreality.cz"), "seznam")


ENGINES = (_brave, _ddg, _seznam, _bing)
SEARCH_HOSTS = ("google.", "bing.", "duckduckgo.", "brave.", "yahoo.", "current.com", "msn.com")


def _cache_get(kind: str, key: str):
    if not config.CACHE_DIR:
        return None
    import hashlib, os
    path = os.path.join(config.CACHE_DIR, kind, hashlib.sha1(key.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _cache_put(kind: str, key: str, value) -> None:
    if not config.CACHE_DIR:
        return
    import hashlib, os
    d = os.path.join(config.CACHE_DIR, kind)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, hashlib.sha1(key.encode()).hexdigest() + ".json"), "w", encoding="utf-8") as f:
        json.dump(value, f)


def _engine_cached(fn, q: str, client: httpx.Client) -> list[SearchHit]:
    key = fn.__name__ + "|" + q
    hit = _cache_get("search", key)
    if hit is not None:
        return [SearchHit(**h) for h in hit]
    res = _safe(fn, q, client)
    if res:
        _cache_put("search", key, [h.__dict__ for h in res])
    return res


def relevant(hit: SearchHit, query_tokens: set[str]) -> bool:
    """Drop the unrelated results engines serve to automated clients."""
    if any(h in domain_of(hit.url) for h in SEARCH_HOSTS):
        return False
    hay = (hit.title + " " + hit.snippet + " " + hit.url).lower()
    return sum(1 for t in query_tokens if t in hay) >= min(2, len(query_tokens))


def search(queries: list[str], per_query: int = 8, timeout_s: float = 8.0) -> list[SearchHit]:
    """Run every query on both engines in parallel and fuse the rankings (RRF).

    Whatever has not answered after `timeout_s` is dropped.
    """
    queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    client = _client()
    pool = ThreadPoolExecutor(max_workers=config.IO_WORKERS)
    import time as _t
    t_end = _t.time() + timeout_s
    # Brave and DuckDuckGo first; Bing only if both come back empty (it tends
    # to serve unrelated results to automated clients).
    jobs = [pool.submit(_engine_cached, fn, q, client) for q in queries for fn in ENGINES[:3]]
    done, _ = wait(jobs, timeout=max(0.5, t_end - _t.time()))
    lists = [j.result() for j in jobs if j in done]
    if not any(lists) and t_end - _t.time() > 1:
        jobs = [pool.submit(_engine_cached, _bing, q, client) for q in queries]
        done, _ = wait(jobs, timeout=max(0.5, t_end - _t.time()))
        lists += [j.result() for j in jobs if j in done]
    pool.shutdown(wait=False, cancel_futures=True)
    qtok = {t for q in queries for t in re.findall(r"[a-z0-9][a-z0-9.+#-]{2,}", q.lower())
            if t not in ("what", "the", "latest", "current", "number", "phone", "version", "which", "who", "how", "for", "and", "with")}
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    for hits in lists:
        for h in hits[:per_query]:
            if not relevant(h, qtok):
                continue
            key = h.url.split("#")[0].rstrip("/")
            if any(d in domain_of(key) for d in LOW_VALUE_DOMAINS):
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + h.rank)
            if key not in best or len(h.snippet) > len(best[key].snippet):
                best[key] = h
    ordered = sorted(scores, key=lambda k: -scores[k])
    return [best[k] for k in ordered]


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return []


# ---------------------------------------------------------------- fetch

_BLOCK_TAGS = ("script", "style", "noscript", "svg", "iframe", "template")


def parse_html(url: str, raw: str) -> Page:
    soup = BeautifulSoup(raw, "lxml")
    page = Page(url=url, ok=True)
    page.title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if isinstance(it, dict):
                page.jsonld.append(it)
                for g in it.get("@graph", []) if isinstance(it.get("@graph"), list) else []:
                    if isinstance(g, dict):
                        page.jsonld.append(g)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        txt = a.get_text(" ", strip=True)
        if href.lower().startswith("tel:"):
            page.tel.append(urllib.parse.unquote(href[4:]))
        elif href.lower().startswith("mailto:"):
            page.mailto.append(urllib.parse.unquote(href[7:].split("?")[0]))
        elif not href.startswith(("#", "javascript:")):
            page.links.append((txt[:80], urllib.parse.urljoin(url, href)))
    for t in soup(_BLOCK_TAGS):
        t.decompose()
    # Keep table rows on one line so "label | value" pairs stay together.
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        tr.replace_with(soup.new_string("\n" + " | ".join(c for c in cells if c) + "\n"))
    text = soup.get_text("\n")
    text = htmllib.unescape(text)
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    page.text = "\n".join(ln for ln in lines if ln)
    return page


def fetch(url: str, client: httpx.Client | None = None) -> Page:
    cached = _cache_get("page", url)
    if cached is not None:
        p = Page(**{k: v for k, v in cached.items() if k != "links"})
        p.links = [tuple(x) for x in cached.get("links", [])]
        return p
    page = _fetch(url, client)
    if page.ok:
        _cache_put("page", url, page.__dict__)
    return page


def _fetch(url: str, client: httpx.Client | None = None) -> Page:
    own = client is None
    client = client or _client()
    try:
        r = client.get(url)
        ctype = r.headers.get("content-type", "")
        if r.status_code >= 400:
            return Page(url=url, error=f"http {r.status_code}")
        if "json" in ctype:
            return Page(url=str(r.url), text=r.text[:200000], ok=True, via="api")
        if "html" not in ctype and "text" not in ctype:
            return Page(url=url, error=f"type {ctype}")
        return parse_html(str(r.url), r.text[:3_000_000])
    except Exception as e:  # network errors are expected; never raise
        return Page(url=url, error=type(e).__name__)
    finally:
        if own:
            client.close()


def fetch_many(urls: list[str], timeout_s: float = 8.0) -> list[Page]:
    """Fetch pages in parallel; pages not back within `timeout_s` are skipped."""
    client = _client()
    pool = ThreadPoolExecutor(max_workers=config.IO_WORKERS)
    jobs = [pool.submit(fetch, u, client) for u in urls]
    done, _ = wait(jobs, timeout=timeout_s)
    pool.shutdown(wait=False, cancel_futures=True)
    return [j.result() if j in done else Page(url=u, error="timeout") for j, u in zip(jobs, urls)]


def get_json(url: str, params: dict | None = None, timeout: float = 4.0) -> dict | list | None:
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    cached = _cache_get("api", key)
    if cached is not None:
        return cached
    try:
        with httpx.Client(headers={"User-Agent": config.API_USER_AGENT}, timeout=timeout,
                          follow_redirects=True) as c:
            r = c.get(url, params=params)
            if r.status_code != 200:
                return None
            data = r.json()
            _cache_put("api", key, data)
            return data
    except Exception:
        return None


FOLLOW_HINTS = {
    "phone": ("contact", "support", "help", "customer", "call", "about"),
    "email": ("contact", "support", "help", "press", "about"),
    "address": ("contact", "location", "about", "visit", "office"),
    "version": ("release", "download", "changelog", "news"),
    "price": ("pricing", "price", "plans", "buy"),
}


def follow_candidates(page: Page, answer_type: str, limit: int) -> list[str]:
    hints = FOLLOW_HINTS.get(answer_type, ())
    if not hints:
        return []
    base = registrable(page.url)
    scored = []
    for txt, href in page.links:
        if registrable(domain_of(href)) != base:
            continue
        low = (txt + " " + href).lower()
        s = sum(2 if h in txt.lower() else 1 for h in hints if h in low)
        if s:
            scored.append((s, href.split("#")[0]))
    seen, out = set(), []
    for _, h in sorted(scored, key=lambda x: -x[0]):
        if h not in seen and h.rstrip("/") != page.url.rstrip("/"):
            seen.add(h)
            out.append(h)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- wikidata

def wikidata_entity(name: str) -> dict | None:
    """Best Wikidata match for a name: id, label, description, official website."""
    if not name:
        return None
    data = get_json("https://www.wikidata.org/w/api.php", {
        "action": "wbsearchentities", "search": name, "language": "en", "format": "json", "limit": 3})
    if not isinstance(data, dict) or not data.get("search"):
        return None
    top = data["search"][0]
    ent = get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{top['id']}.json")
    site = ""
    try:
        claims = ent["entities"][top["id"]]["claims"]
        best = None
        for c in claims.get("P856", []):
            v = c["mainsnak"]["datavalue"]["value"]
            if c.get("rank") == "deprecated":
                continue
            host = urllib.parse.urlparse(v).netloc.lower()
            sub = host.split(".")[0]
            score = (2 if c.get("rank") == "preferred" else 0) + (1 if sub == "www" or host.count(".") == 1 else 0)                 - (1 if len(sub) == 2 and sub != "go" else 0)
            if best is None or score > best[0]:
                best = (score, v)
        if best:
            site = best[1]
    except Exception:
        pass
    return {"id": top["id"], "label": top.get("label", ""), "description": top.get("description", ""), "website": site}
