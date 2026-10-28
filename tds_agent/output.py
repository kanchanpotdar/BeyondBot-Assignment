"""Dependency-free early outputs, full redacted trace, validated checkpoints."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

INCOMPLETE = "Incomplete run: financial coverage has not been verified."


def exact_json(value, **kwargs):
    """Emit Decimal as JSON number text, never via binary floating point."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite Decimal cannot be serialized")
        return format(value, "f")
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + exact_json(v) for k, v in value.items()) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(exact_json(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def strict_json(text):
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON property: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_float=Decimal, parse_constant=reject_constant, object_pairs_hook=unique_keys)


class Output:
    def __init__(self, out_dir):
        self.directory = Path(out_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.secrets = tuple(sorted({v for key in ("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
                                     if (v := os.environ.get(key))}, key=len, reverse=True))
        self.trace_path = self.directory / "trace.jsonl"
        self.trace_path.touch(exist_ok=True)
        self.validator = None
        self.financial = None
        self.state = {"findings": [], "limitations": [INCOMPLETE],
                      "run": {"run_id": self.run_id, "status": "initializing", "coverage_complete": False}}
        self._write(self.state)
        self.trace("run_started", phase=1)

    def clean(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self.clean(str(k)): "[REDACTED]" if str(k).lower() in
                    {"api_key", "authorization", "access_token", "secret"} else self.clean(v)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        return value

    def trace(self, event, **fields):
        record = self.clean({"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                             "event": event, **fields})
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(exact_json(record) + "\n")
            handle.flush()

    def error(self, exc):
        return {"type": type(exc).__name__, "message": self.clean(str(exc))}

    def attach_schema(self, schema_path):
        from jsonschema import Draft202012Validator

        schema = strict_json(Path(schema_path).read_text())
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)
        self.validator.validate(self.state)

    def _write(self, state):
        # Serialize first: even serialization failure cannot replace a good result.
        payload = exact_json(self.clean(state)) + "\n"
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=".findings-", suffix=".tmp", delete=False) as handle:
                path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(path, self.directory / "findings.json")
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def checkpoint(self, candidate):
        if self.validator is None:
            raise RuntimeError("Schema validation has not initialized")
        candidate = self.clean(copy.deepcopy(candidate))
        self.validator.validate(candidate)
        self._write(candidate)
        self.state = candidate
        self.trace("checkpoint", findings=len(candidate["findings"]))

    def finalize(self, reason, metrics=None, limitation=None):
        # Only trusted, type-controlled fields are added; no dependency import needed
        # after a failed setup. Findings remain the last validated checkpoint.
        candidate = self.financial.snapshot() if self.financial else copy.deepcopy(self.state)
        candidate.setdefault("limitations", [])
        for text in ((None if self.financial else INCOMPLETE), limitation):
            if text and str(text) not in candidate["limitations"]:
                candidate["limitations"].append(str(text))
        if metrics and metrics.get("model_attempts") and metrics.get("cost_usd") is None:
            candidate["limitations"].append("API cost metadata was unavailable for one or more attempts; actual cost and enforcement of a reported-cost cap cannot be guaranteed.")
        complete = bool(self.financial and self.financial.data.coverage_report()["complete"])
        candidate["run"] = {**candidate.get("run", {}), **(metrics or {}), "run_id": self.run_id,
                            "status": "complete" if complete else "incomplete", "termination_reason": str(reason),
                            "coverage_complete": complete, "seconds": round(time.monotonic() - self.started, 3)}
        candidate = self.clean(candidate)
        if self.validator is not None:
            self.validator.validate(candidate)
        self._write(candidate)
        self.state = candidate
        if self.financial:
            self.trace("coverage_final", coverage=self.financial.data.coverage_report(),
                       unprocessed_vouchers=[v for v, s in self.financial.data.coverage.items() if s == "unprocessed"],
                       unresolved_vouchers=[v for v, s in self.financial.data.coverage.items() if s == "unresolved"])
        self.trace("run_finished", reason=reason, metrics=candidate["run"])
