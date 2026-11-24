"""Tests for the skill-eval scorecard UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_benchmark

SKILL = {
    "skill_name": "csv-cleaner",
    "description": "Cleans and standardizes messy CSV files",
    "instructions": "When asked to clean a CSV, run the pipeline.",
}
SCENARIOS = [
    {"prompt": "Clean up this CSV file", "expected_trigger": True, "scenario_type": "positive"},
    {"prompt": "Tell me a joke", "expected_trigger": False, "scenario_type": "negative"},
]


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/benchmark",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_benchmark_endpoint_runs_real_evaluator(server):
    status, payload = _post(server, {**SKILL, "scenarios": SCENARIOS})
    assert status == 200
    assert payload["total_scenarios"] == 2
    assert len(payload["results"]) == 2
    for key in ("accuracy_pct", "precision_pct", "recall_pct", "f1_score", "safety_score"):
        assert key in payload


def test_result_is_json_serializable():
    result = run_benchmark(**SKILL, scenarios=SCENARIOS)
    json.dumps(result)  # must not raise


def test_health_and_index_served(server):
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert health["scenario_types"] == ["positive", "negative", "adversarial", "boundary"]
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "scorecard" in html and "/api/benchmark" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({**SKILL, "scenarios": []}, "scenarios must be a non-empty list"),
        ({**SKILL, "skill_name": "  "}, "skill_name must be a non-empty string"),
        ({**SKILL, "scenarios": [{"prompt": "x"}]}, "expected_trigger must be a boolean"),
        ({**SKILL, "scenarios": [{"prompt": "x", "expected_trigger": True, "scenario_type": "magic"}]}, "scenario_type must be one of"),
        ({**SKILL, "scenarios": [{"prompt": "  ", "expected_trigger": True}]}, "prompt must be a non-empty string"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"
