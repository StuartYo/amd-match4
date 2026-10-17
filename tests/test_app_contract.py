"""Offline tests for the container entry point (app.py) and the resident server.

No network, no model: agent.answer is replaced by a stub.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from webagent import agent, llm, server, web  # noqa: E402

spec = importlib.util.spec_from_file_location("wa_app", ROOT / "app" / "app.py")
appmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appmod)


class Fake:
    def __init__(self, answer="42", delay=0.0, boom=False):
        self.answer, self.delay, self.boom, self.calls = answer, delay, boom, 0

    def __call__(self, q, budget=None):
        self.calls += 1
        time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("boom")
        return agent.Result(self.answer, 0.9, ["https://example.com"], "text", {"q": q})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "READY_FILE", tmp_path / "ready")
    monkeypatch.setattr(appmod, "STARTING_FILE", tmp_path / "starting")
    monkeypatch.setattr(appmod.config, "SERVER_PORT", 59999)  # nothing listens here
    return tmp_path


def run(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py"] + args)
    appmod.main()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def test_query_form_writes_contract_keys(env, monkeypatch):
    fake = Fake("800-221-1212")
    monkeypatch.setattr(agent, "answer", fake)
    run(["--query-id", "query_01", "--query", "q?", "--output-dir", str(env / "out")], monkeypatch)
    out = read(env / "out" / "query_01_output.json")
    assert out["answer"] == "800-221-1212" and out["sources"] == ["https://example.com"]
    assert {"answer", "text", "sources", "citations", "confidence"} <= set(out)
    assert not any(k.startswith("_") for k in out)


def test_positional_and_question_flag(env, monkeypatch):
    monkeypatch.setattr(agent, "answer", Fake("a"))
    run(["hello?", "--output-dir", str(env / "o1")], monkeypatch)
    assert read(env / "o1" / "answer_output.json")["answer"] == "a"
    run(["--question", "hello?", "--query-id", "x y/z", "--output-dir", str(env / "o2")], monkeypatch)
    assert read(env / "o2" / "x_y_z_output.json")["answer"] == "a"  # unsafe characters cleaned


def test_input_file_single_and_list(env, monkeypatch):
    monkeypatch.setattr(agent, "answer", Fake("v"))
    single = env / "q_01.json"
    single.write_text(json.dumps({"id": "ignored", "question": "q?"}), encoding="utf-8")
    run(["--input", str(single), "--output-dir", str(env / "o")], monkeypatch)
    assert read(env / "o" / "q_01_output.json")["answer"] == "v"
    many = env / "batch.json"
    many.write_text(json.dumps([{"query_id": "a1", "query": "x"}, {"id": "b2", "question": "y"}]), encoding="utf-8")
    run(["--input", str(many), "--output-dir", str(env / "o")], monkeypatch)
    assert (env / "o" / "a1_output.json").exists() and (env / "o" / "b2_output.json").exists()


def test_input_dir(env, monkeypatch):
    monkeypatch.setattr(agent, "answer", Fake("d"))
    d = env / "in"
    d.mkdir()
    (d / "q_02.txt").write_text("plain text question", encoding="utf-8")
    (d / "q_03.json").write_text('{"query": "json question"}', encoding="utf-8")
    run(["--input-dir", str(d), "--output-dir", str(env / "o")], monkeypatch)
    assert read(env / "o" / "q_02_output.json")["answer"] == "d"
    assert read(env / "o" / "q_03_output.json")["answer"] == "d"


def test_unknown_flags_are_tolerated(env, monkeypatch):
    monkeypatch.setattr(agent, "answer", Fake("k"))
    run(["--query", "q", "--query-id", "u", "--timeout", "30", "--output-dir", str(env / "o")], monkeypatch)
    assert read(env / "o" / "u_output.json")["answer"] == "k"


def test_agent_crash_leaves_valid_empty_file(env, monkeypatch):
    monkeypatch.setattr(agent, "answer", Fake(boom=True))
    run(["--query-id", "c", "--query", "q", "--output-dir", str(env / "o")], monkeypatch)
    out = read(env / "o" / "c_output.json")
    assert out["answer"] == "" and out["sources"] == []


def test_hard_deadline_ends_a_hung_agent(env, monkeypatch):
    monkeypatch.setattr(agent, "answer", Fake("late", delay=6.0))
    monkeypatch.setattr(appmod.config, "HARD_DEADLINE_S", 1.0)
    t = time.time()
    run(["--query-id", "h", "--query", "q", "--output-dir", str(env / "o")], monkeypatch)
    assert time.time() - t < 3.0
    assert read(env / "o" / "h_output.json")["answer"] == ""


def test_starting_server_that_never_answers_is_not_replaced_by_in_process_model(env, monkeypatch):
    fake = Fake("should-not-run")
    monkeypatch.setattr(agent, "answer", fake)
    (env / "starting").write_text("1")
    monkeypatch.setattr(appmod.config, "SERVER_WAIT_S", 0.5)
    run(["--query-id", "s", "--query", "q", "--output-dir", str(env / "o")], monkeypatch)
    assert fake.calls == 0
    assert read(env / "o" / "s_output.json")["answer"] == ""


def test_server_budget_and_loading(monkeypatch):
    # model still loading -> empty answer within the budget, not a hang
    server.READY.clear()
    t = time.time()
    out = server.handle_question("q", 3.0)
    assert out["answer"] == "" and time.time() - t < 3.5
    # ready + hung agent -> empty answer shortly after the budget
    server.READY.set()
    monkeypatch.setattr(agent, "answer", Fake("late", delay=6.0))
    t = time.time()
    out = server.handle_question("q", 2.0)
    assert out["answer"] == "" and time.time() - t < 4.5
    # ready + good agent
    monkeypatch.setattr(agent, "answer", Fake("ok"))
    assert server.handle_question("q", 5.0)["answer"] == "ok"


class DeadLLM(llm.BaseLLM):
    def chat(self, *a, **k):
        raise RuntimeError("no model")


def _offline_web(monkeypatch, text):
    hit = web.SearchHit("https://www.acme.com/contact", "Acme contact", "Call Acme at 800-555-0123", "fake", 1)
    monkeypatch.setattr(web, "search", lambda qs, **k: [hit])
    monkeypatch.setattr(web, "wikidata_entity", lambda e: None)
    monkeypatch.setattr(web, "fetch_many", lambda urls, **k: [web.Page(url=u, ok=True, title="Contact", text=text) for u in urls])


def test_agent_still_answers_when_the_model_is_dead(monkeypatch):
    """Degraded mode: model load/chat failures must not stop the deterministic extractors."""
    llm.set_llm(DeadLLM())
    _offline_web(monkeypatch, "Contact Acme. Customer service phone: 800-555-0123. Mon-Fri 9-5.")
    try:
        res = agent.answer("What is the customer service phone number of Acme?", 30)
    finally:
        llm.set_llm(None)
    assert res.answer == "800-555-0123"
    assert res.sources and res.sources[0].startswith("https://www.acme.com")


def test_unsupported_value_is_left_empty(monkeypatch):
    """A question whose answer is on no fetched page gets an empty answer, not a guess."""
    llm.set_llm(DeadLLM())
    _offline_web(monkeypatch, "Welcome to Acme. We make anvils.")
    monkeypatch.setattr(web, "search", lambda qs, **k: [])
    try:
        res = agent.answer("What is the customer service phone number of Acme?", 30)
    finally:
        llm.set_llm(None)
    assert res.answer == ""


def test_guess_entity_fallback():
    assert agent.guess_entity("What is the latest LTS version of Node.js?") == "Node.js"
    assert agent.guess_entity("Who is the current CEO of Nike?") == "Nike"
    assert agent.guess_entity("What is the USPTO's general phone number?") == "USPTO"
    assert agent.guess_entity("What is the phone number for Grand Canyon National Park?") == "Grand Canyon National Park"
    assert agent.guess_entity("what is the weather") == ""
