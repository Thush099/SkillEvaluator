"""Lightweight local web UI server for the SkillEvaluator quickstart benchmark.

Serves the eval scorecard in ``ui/index.html`` and exposes a JSON API that runs the real
``SimulatedSkillEvaluator`` benchmark from ``scripts/quickstart_eval.py``. Standard library
only. Start with ``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_quickstart():
    path = REPO_ROOT / "scripts" / "quickstart_eval.py"
    spec = importlib.util.spec_from_file_location("quickstart_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_qe = _load_quickstart()

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

SCENARIO_TYPES = ("positive", "negative", "adversarial", "boundary")
MAX_TEXT = 20_000


def run_benchmark(
    skill_name: str,
    description: str,
    instructions: str,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the real skill-evaluation benchmark and return the report.

    Raises ValueError with a clear message for invalid inputs.
    """
    if not isinstance(skill_name, str) or not skill_name.strip():
        raise ValueError("skill_name must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")
    if len(scenarios) > 500:
        raise ValueError("scenarios must contain at most 500 entries")

    parsed = []
    for index, entry in enumerate(scenarios):
        if not isinstance(entry, dict):
            raise ValueError(f"scenario[{index}] must be an object")
        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"scenario[{index}].prompt must be a non-empty string")
        if len(prompt) > MAX_TEXT:
            raise ValueError(f"scenario[{index}].prompt exceeds {MAX_TEXT} characters")
        stype = entry.get("scenario_type", "positive")
        if stype not in SCENARIO_TYPES:
            raise ValueError(
                f"scenario[{index}].scenario_type must be one of {list(SCENARIO_TYPES)}, got {stype!r}"
            )
        expected = entry.get("expected_trigger")
        if not isinstance(expected, bool):
            raise ValueError(f"scenario[{index}].expected_trigger must be a boolean")
        parsed.append(
            _qe.EvalScenario(
                prompt=prompt,
                expected_trigger=expected,
                scenario_type=stype,
                notes=str(entry.get("notes", "")),
            )
        )

    evaluator = _qe.SimulatedSkillEvaluator(skill_name.strip(), description, instructions)
    report = evaluator.run_benchmark(parsed)
    return {
        "skill_name": report.skill_name,
        "timestamp": report.timestamp,
        "total_scenarios": report.total_scenarios,
        "passed_scenarios": report.passed_scenarios,
        "accuracy_pct": round(report.accuracy_pct, 3),
        "precision_pct": round(report.precision_pct, 3),
        "recall_pct": round(report.recall_pct, 3),
        "f1_score": round(report.f1_score, 4),
        "safety_score": round(report.safety_score, 3),
        "results": report.results,
    }


MAX_REQUEST_BYTES = 4_000_000


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok", "scenario_types": list(SCENARIO_TYPES)})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/benchmark":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_REQUEST_BYTES:
                    self._send_json(
                        {"error": f"request body exceeds {MAX_REQUEST_BYTES} bytes"},
                        status=413,
                    )
                    return
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_benchmark(
                    skill_name=payload.get("skill_name", ""),
                    description=payload.get("description", ""),
                    instructions=payload.get("instructions", ""),
                    scenarios=payload.get("scenarios", []),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the skill-eval scorecard UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8131)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Skill-eval scorecard running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())