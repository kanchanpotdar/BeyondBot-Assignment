"""Audit persisted Phase 1 outputs without invoking any model or data tools."""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = Path(args.out)
    findings = json.loads((directory / "findings.json").read_text())
    schema = json.loads((root / "findings.schema.json").read_text())
    Draft202012Validator(schema).validate(findings)
    records = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    assert records and all(r["ts"] and r["run_id"] and r["event"] for r in records)
    current = [r for r in records if r["run_id"] == findings["run"]["run_id"]]
    attempts = [r for r in current if r["event"] == "model_attempt"]
    terminals = [r for r in current if r["event"] in ("model_response", "model_error")]
    assert [r["attempt"] for r in attempts] == [r["attempt"] for r in terminals]
    for attempt in attempts:
        assert attempt["messages"] and attempt["tools"]
        pending = []
        for message in attempt["messages"]:
            if message["role"] == "assistant":
                assert not pending
                pending = [c["id"] for c in message.get("tool_calls") or []]
            elif message["role"] == "tool":
                assert message["tool_call_id"] in pending
                json.loads(message["content"])
                pending.remove(message["tool_call_id"])
        assert not pending
    starts = [r for r in current if r["event"] == "tool_start"]
    results = [r for r in current if r["event"] == "tool_result"]
    assert [r["tool_call_id"] for r in starts] == [r["tool_call_id"] for r in results]
    assert all("arguments" in r for r in starts)
    assert all("result" in r and "arguments" in r for r in results)
    assert current[-1]["event"] == "run_finished"
    assert not findings["run"]["coverage_complete"]
    secrets = [value for key, value in os.environ.items() if value and any(
        part in key.upper() for part in ("API_KEY", "ACCESS_TOKEN", "PASSWORD", "CLIENT_SECRET"))]
    secrets += ["phase1-local-test-token", "phase1-synthetic-secret-canary", "private-test-key"]
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".log")]
    for path in files:
        text = path.read_text()
        assert all(secret not in text for secret in secrets), f"Credential canary detected in {path.name}"
        assert not re.search(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{16,}", text), f"Key-like string in {path.name}"
        assert not re.search(r"Bearer\s+[A-Za-z0-9._-]{12,}", text), f"Bearer credential in {path.name}"
    print(json.dumps({"schema_valid": True, "trace_valid": True, "model_attempts": len(attempts),
                      "tool_calls": len(starts), "tool_errors": dict(Counter(
                          r["result"].get("error_code") for r in results if not r["result"]["success"])),
                      "termination_reason": findings["run"]["termination_reason"],
                      "credential_scan_files": len(files), "credential_scan_passed": True}, indent=2))


if __name__ == "__main__":
    main()
