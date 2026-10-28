"""Read-only schema/trace/provenance audit, including nested evidence calls."""
import argparse
import json
import os
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

from jsonschema import Draft202012Validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = Path(args.out)
    parse = lambda text: json.loads(text, parse_float=Decimal)
    findings = parse((folder / "findings.json").read_text())
    Draft202012Validator(parse((root / "findings.schema.json").read_text())).validate(findings)
    all_records = [parse(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    records = [r for r in all_records if r["run_id"] == findings["run"]["run_id"]]
    assert records and all(r["ts"] for r in records)
    pending, seen, completed = {}, set(), set()
    for record in records:
        event = record["event"]
        if event == "tool_start":
            identity = record["tool_call_id"]
            assert identity not in seen, "Duplicate tool-call identity"
            assert "arguments" in record
            pending[identity] = record
            seen.add(identity)
        elif event == "tool_result":
            identity = record["tool_call_id"]
            assert identity in pending, "Tool result has no matching start"
            assert pending.pop(identity)["name"] == record["name"]
            assert "result" in record and "arguments" in record
            completed.add(identity)
        elif event == "model_attempt":
            waiting = set()
            for message in record["messages"]:
                if message["role"] == "assistant":
                    assert not waiting
                    waiting.update(c["id"] for c in message.get("tool_calls") or [])
                elif message["role"] == "tool":
                    assert message["tool_call_id"] in waiting
                    waiting.remove(message["tool_call_id"])
                    parse(message["content"])
            assert not waiting
    assert not pending
    attempts = Counter(r["attempt"] for r in records if r["event"] == "model_attempt")
    terminals = Counter(r["attempt"] for r in records if r["event"] in ("model_response", "model_error"))
    assert attempts == terminals
    coverage = next(r for r in reversed(records) if r["event"] == "coverage_final")
    assert findings["run"]["coverage_complete"] == coverage["coverage"]["complete"]
    assert records[-1]["event"] == "run_finished"
    secrets = [v for k, v in os.environ.items() if len(v) >= 8 and any(term in k.upper() for term in ("API_KEY", "ACCESS_TOKEN", "PASSWORD", "CLIENT_SECRET"))]
    secrets += ["phase1-local-test-token", "phase1-synthetic-secret-canary"]
    for path in folder.glob("*"):
        if path.suffix not in (".json", ".jsonl", ".log"):
            continue
        text = path.read_text()
        assert not any(secret in text for secret in secrets), "Configured credential/canary in output"
        assert not re.search(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{12,}", text)
    print(json.dumps({"schema_valid": True, "trace_valid": True, "tool_calls": len(completed),
                      "model_attempts": sum(attempts.values()), "findings": len(findings["findings"]),
                      "coverage_complete": findings["run"]["coverage_complete"],
                      "unprocessed": len(coverage["unprocessed_vouchers"]),
                      "unresolved": len(coverage["unresolved_vouchers"]),
                      "secret_scan_passed": True, "real_model_performance_verified": False}, indent=2))


if __name__ == "__main__":
    main()
