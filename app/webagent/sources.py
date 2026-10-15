"""Authoritative structured sources tried before free-text search.

These answer "latest version" questions straight from package registries and
endoflife.date, which are more current than any model's memory and less noisy
than blog posts.
"""
from __future__ import annotations

import datetime as dt
import re

from .extract import Candidate
from .web import get_json

_EOL_PRODUCTS: list[str] | None = None


def _eol_products() -> list[str]:
    global _EOL_PRODUCTS
    if not _EOL_PRODUCTS:  # do not cache a failed fetch
        for _ in range(2):
            data = get_json("https://endoflife.date/api/all.json")
            if isinstance(data, list) and data:
                _EOL_PRODUCTS = data
                break
    return _EOL_PRODUCTS or []


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


EOL_ALIASES = {
    "node": "nodejs", "node-js": "nodejs", "nodejs": "nodejs", "k8s": "kubernetes",
    "golang": "go", "postgres": "postgresql", "dotnet": "dotnet", "net": "dotnet",
    "java": "oracle-jdk", "openjdk": "openjdk-builds-from-oracle", "mac-os": "macos", "macos": "macos",
    "windows-11": "windows", "ruby-on-rails": "rails", "vue-js": "vue", "vuejs": "vue",
    "react-js": "react", "angular-js": "angularjs", "ms-sql": "mssqlserver", "sql-server": "mssqlserver",
}


def eol_product_for(entity: str) -> str | None:
    s = _slug(entity)
    s = EOL_ALIASES.get(s, s)
    prods = _eol_products()
    if not prods:  # list unavailable: try the slug itself
        return s if get_json(f"https://endoflife.date/api/{s}.json") else None
    if s in prods:
        return s
    s2 = s.replace("-", "")
    for p in prods:
        if p.replace("-", "") == s2:
            return p
    return None


def _is_lts(value, today: dt.date) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value) <= today
        except ValueError:
            return False
    return False


def latest_from_eol(product: str, want_lts: bool) -> Candidate | None:
    data = get_json(f"https://endoflife.date/api/{product}.json")
    if not isinstance(data, list) or not data:
        return None
    today = dt.date.today()
    cycles = data
    if want_lts:
        cycles = [c for c in data if _is_lts(c.get("lts"), today)]
        if not cycles:
            return None
    for c in cycles:
        rel = c.get("releaseDate")
        if isinstance(rel, str):
            try:
                if dt.date.fromisoformat(rel) > today:
                    continue
            except ValueError:
                pass
        latest = c.get("latest")
        if latest:
            return Candidate(str(latest), f"https://endoflife.date/{product}", "api",
                             context=f"endoflife.date cycle {c.get('cycle')} latest {latest}" + (" (LTS)" if want_lts else ""))
    return None


def latest_from_pypi(name: str) -> Candidate | None:
    data = get_json(f"https://pypi.org/pypi/{name}/json")
    if isinstance(data, dict) and data.get("info", {}).get("version"):
        v = data["info"]["version"]
        return Candidate(v, f"https://pypi.org/project/{name}/", "api", context=f"PyPI latest {v}")
    return None


def latest_from_npm(name: str) -> Candidate | None:
    data = get_json(f"https://registry.npmjs.org/{name}/latest")
    if isinstance(data, dict) and data.get("version"):
        v = data["version"]
        return Candidate(v, f"https://www.npmjs.com/package/{name}", "api", context=f"npm latest {v}")
    return None


def latest_from_github(repo: str) -> Candidate | None:
    data = get_json(f"https://api.github.com/repos/{repo}/releases/latest")
    if isinstance(data, dict) and data.get("tag_name"):
        tag = data["tag_name"]
        m = re.search(r"\d+(?:\.\d+)+(?:[-.]?\w+)?", tag)
        v = m.group(0) if m else tag
        return Candidate(v, f"https://github.com/{repo}/releases/latest", "api", context=f"GitHub latest release {tag}")
    return None


def version_candidates(plan: dict, github_repos: list[str]) -> list[Candidate]:
    """Only used when the question asks for the latest/current version."""
    out: list[Candidate] = []
    entity = plan.get("entity") or ""
    q = (plan.get("question") or "").lower()
    want_lts = "lts" in q or "long-term support" in q or "long term support" in q
    eco = (plan.get("ecosystem") or "").lower()
    pkg = plan.get("package") or ""
    prod = eol_product_for(pkg or entity) if entity or pkg else None
    if prod:
        c = latest_from_eol(prod, want_lts)
        if c:
            out.append(c)
    if want_lts:
        return out  # registries do not know about LTS lines
    if pkg and eco in ("pypi", "python", "pip"):
        c = latest_from_pypi(pkg)
        if c:
            out.append(c)
    if pkg and eco in ("npm", "javascript", "node"):
        c = latest_from_npm(pkg)
        if c:
            out.append(c)
    for repo in github_repos[:1]:
        c = latest_from_github(repo)
        if c:
            out.append(c)
    return out
