"""Offline tests: no network, no model."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from webagent import agent, extract, llm, web  # noqa: E402


def test_grade_norm_matches_grader():
    assert extract.grade_norm("7-abc 123") == "7ABC123"
    assert extract.grade_norm("A·12345") == "A12345"


def test_phone_formatting():
    assert extract.format_phone("(800) 221-1212") == "800-221-1212"
    assert extract.format_phone("1-800-221-1212") == "800-221-1212"
    assert extract.format_phone("+1 800 221 1212") == "800-221-1212"
    assert extract.format_phone("+44 (0)20 7946 0958") == "+44 020 7946 0958"


def test_phone_not_year_range_or_date():
    assert not extract.looks_like_phone("2019-2024")
    assert not extract.looks_like_phone("10/15/2026")


def test_phone_grounding_needs_a_real_number():
    text = "Call 800-221-1212 now. Order 4002 211 212 items."
    assert extract.value_in_text("800-221-1212", text, "phone")
    assert not extract.value_in_text("800-221-1213", text, "phone")


def test_version_extraction():
    vals = [v for v, _ in extract.find_typed("Node.js v24.22.0 (LTS) and 26.12.0 Current", "version")]
    assert "24.22.0" in vals and "26.12.0" in vals
    assert extract.format_answer("v1.37.1", "version") == "1.37.1"


def test_parse_json_from_chatty_reply():
    assert llm.parse_json('Sure! ```json\n{"answer": "5.3.0", "source": 2}\n```') == {"answer": "5.3.0", "source": 2}
    assert llm.parse_json("no json here") is None


def test_rule_type():
    assert agent.rule_type("What is the IRS phone number?") == "phone"
    assert agent.rule_type("What is the latest LTS version of Node.js?") == "version"
    assert agent.rule_type("Who is the CEO of Nike?") == "text"


def test_html_parsing_keeps_tel_and_tables():
    raw = """<html><head><title>Contact</title>
    <script type="application/ld+json">{"@type":"Organization","telephone":"+1-800-555-0100"}</script></head>
    <body><a href="tel:+18005550100">Call us</a><table><tr><td>Version</td><td>2.1.0</td></tr></table>
    <a href="/support">Support</a></body></html>"""
    p = web.parse_html("https://example.com/x", raw)
    assert p.tel == ["+18005550100"]
    assert p.jsonld[0]["telephone"] == "+1-800-555-0100"
    assert "Version | 2.1.0" in p.text
    assert ("Support", "https://example.com/support") in p.links


class FakeLLM(llm.BaseLLM):
    def chat(self, messages, max_tokens=256, usage=None, **kw):
        return '{"answer": "999-999-9999", "source": 1}'


def test_ungrounded_llm_answer_is_rejected():
    llm.set_llm(FakeLLM())
    page = web.Page(url="https://example.com", ok=True, text="Call us at 800-555-0100 for help.")
    plan = {"question": "phone number of example", "answer_type": "phone", "entity": "example", "attribute": "phone"}
    cands = [extract.Candidate("999-999-9999", page.url, "llm"), extract.Candidate("800-555-0100", page.url, "regex", "Call us at")]
    ranked = agent.score_candidates(cands, [page], plan)
    assert [c.value for c in ranked] == ["800-555-0100"]
    llm.set_llm(None)


def test_app_writes_placeholder_then_answer(tmp_path):
    # --input with a json file; the agent is replaced by an env flag-free stub server-less run would
    # need network, so only check argument parsing and the placeholder writer here.
    sys.path.insert(0, str(ROOT / "app"))
    import app as appmod
    out = tmp_path / "q_output.json"
    appmod.write(out, appmod.EMPTY)
    assert json.loads(out.read_text()) == appmod.EMPTY
    q = tmp_path / "q_01.json"
    q.write_text(json.dumps({"question": "What is X?"}))
    assert appmod.read_question(q) == "What is X?"
