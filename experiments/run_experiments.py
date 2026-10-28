#!/usr/bin/env python3
"""Safe-by-default five-run experiment preparation. No .env loading or model client.

Default invocation prints a plan only. Execution requires --execute and either
--simulation (localhost only, never scored) or --allow-real (explicit future use).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tds_agent.output import exact_json, strict_json  # standard-library-only import

RUNS = {**{f"baseline_{i}": "agent" for i in range(1, 4)},
        **{f"variant_{i}": "proactive" for i in range(1, 3)}}
NOT_RUN = "NOT RUN — API ACCESS UNAVAILABLE"


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(exact_json(value) + "\n")


def fingerprint(data):
    """Public financial inputs/code only; protected backing files are not read."""
    files = [data / "client.json", *sorted((data / "books").glob("*.csv")),
             *sorted((data / "documents").glob("*.txt")), *sorted((data / "rules").glob("*.yaml")),
             ROOT / "run.py", ROOT / "findings.schema.json", ROOT / "score.py", *sorted((ROOT / "tds_agent").glob("*.py"))]
    files.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_for_execution(simulation, allow_real, environment=None):
    env = dict(os.environ if environment is None else environment)
    base, model, key = (env.get(k) for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"))
    if not base or not model or not key:
        raise ValueError("Explicit LLM_BASE_URL, LLM_MODEL and LLM_API_KEY environment variables are required; no default provider or .env fallback")
    url = urlsplit(base)
    if url.username or url.password or url.query or url.fragment or url.scheme not in ("http", "https"):
        raise ValueError("Endpoint must have a supported scheme and no embedded credentials/query")
    local = url.hostname in ("127.0.0.1", "localhost", "::1")
    if simulation and not local:
        raise ValueError("Simulation may use only a localhost fake endpoint")
    if not simulation and not allow_real:
        raise ValueError("Real execution is disabled without explicit --allow-real authorization")
    if not simulation and local:
        raise ValueError("Local fake runs must be labeled --simulation")
    if simulation:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    return env


def redact(text, env):
    for key, value in env.items():
        if value and any(term in key.upper() for term in ("API_KEY", "ACCESS_TOKEN", "PASSWORD", "CLIENT_SECRET")):
            text = text.replace(value, "[REDACTED]")
    return text


def measured_artifacts(folder):
    """Counts come from actual trace events; missing usage is unavailable, not zero."""
    metrics = {"schema_valid": False, "trace_valid": False, "model_attempts": None, "model_calls": None,
               "native_tool_calls": None, "all_traced_tool_calls": None, "prompt_tokens": None,
               "completion_tokens": None, "cost_usd": None, "termination_reason": None, "coverage_complete": False}
    findings = None
    errors = []
    try:
        from jsonschema import Draft202012Validator
        findings = strict_json((folder / "findings.json").read_text())
        Draft202012Validator(strict_json((ROOT / "findings.schema.json").read_text())).validate(findings)
        metrics.update(schema_valid=True, termination_reason=findings.get("run", {}).get("termination_reason"),
                       coverage_complete=findings.get("run", {}).get("coverage_complete", False))
    except Exception as exc:
        errors.append("Findings unavailable/invalid: " + type(exc).__name__)
    try:
        records = [strict_json(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
        run_id = findings.get("run", {}).get("run_id") if findings else None
        if run_id:
            records = [r for r in records if r.get("run_id") == run_id]
        attempts = [r for r in records if r["event"] == "model_attempt"]
        responses = [r for r in records if r["event"] == "model_response"]
        tool_calls = [r for r in records if r["event"] == "tool_start"]
        metrics.update(trace_valid=bool(records), model_attempts=len(attempts), model_calls=len(responses),
                       all_traced_tool_calls=len(tool_calls), native_tool_calls=sum(not r["tool_call_id"].startswith("ingestion_") for r in tool_calls))
        usage = [r["response"].get("usage") or {} for r in responses]
        complete = bool(attempts) and len(responses) == len(attempts)
        for key in ("prompt_tokens", "completion_tokens", "cost"):
            values = [u.get(key) for u in usage]
            if complete and all(isinstance(v, (int, Decimal)) and not isinstance(v, bool) and v >= 0 for v in values):
                metrics["cost_usd" if key == "cost" else key] = sum(values, Decimal(0))
        final_coverage = [r for r in records if r["event"] == "coverage_final"]
        metrics["coverage"] = final_coverage[-1]["coverage"] if final_coverage else None
    except Exception as exc:
        errors.append("Trace unavailable/invalid: " + type(exc).__name__)
    metrics["artifact_errors"] = errors
    return metrics, findings


def finding_differences(runs):
    available = [(name, result["findings"]) for name, result in runs.items() if result.get("findings") is not None]
    differences = []
    for (left, a), (right, b) in itertools.combinations(available, 2):
        ca, cb = Counter(exact_json(f) for f in a), Counter(exact_json(f) for f in b)
        differences.append({"left": left, "right": right, "only_left": [strict_json(x) for x in sorted((ca - cb).elements())],
                            "only_right": [strict_json(x) for x in sorted((cb - ca).elements())]})
    return differences


def execute_runs(data, destination, names, env, simulation, budgets, invoke=subprocess.run):
    # Preflight all destinations before launching any subprocess.
    if any((destination / name).exists() for name in names):
        raise FileExistsError("Selected run directory already exists; prior artifacts will not be overwritten")
    cohort_fingerprint = fingerprint(data)
    results = {}
    for name in names:
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(ROOT / "run.py"), "--data", str(data), "--out", str(folder),
                   "--retrieval-mode", RUNS[name], *budgets]
        configuration = {"name": name, "retrieval_mode": RUNS[name], "model": env["LLM_MODEL"],
                         "base_url": env["LLM_BASE_URL"],
                         "public_input_code_fingerprint": cohort_fingerprint, "simulation_only": simulation,
                         "started_utc": datetime.now(timezone.utc).isoformat(), "command": command}
        write_new(folder / "configuration.json", configuration)
        start, returncode, operational = time.monotonic(), None, None
        try:
            proc = invoke(command, cwd=ROOT, env=env, text=True, capture_output=True)
            returncode = proc.returncode
            (folder / "stdout.log").write_text(redact(proc.stdout or "", env))
            (folder / "stderr.log").write_text(redact(proc.stderr or "", env))
        except KeyboardInterrupt:
            operational = "INTERRUPTED"
        except Exception as exc:
            operational = "EXECUTION_ERROR:" + type(exc).__name__
        duration = time.monotonic() - start
        metrics, findings = measured_artifacts(folder)
        metrics.update(wall_seconds=duration, exit_code=returncode, simulation_only=simulation)
        metrics["status"] = operational or ("COMPLETED" if returncode == 0 and metrics["coverage_complete"] else "INCOMPLETE")
        metrics["score"] = None
        metrics["scoring_status"] = "NOT SCORED — SIMULATION ONLY" if simulation else "NOT SCORED — INVALID OUTPUT"
        if not simulation and metrics["schema_valid"] and operational is None:
            try:
                scored = invoke([sys.executable, str(ROOT / "score.py"), str(folder / "findings.json"), "--json"],
                                cwd=ROOT, env=env, text=True, capture_output=True)
                (folder / "score.stdout.log").write_text(redact(scored.stdout or "", env))
                (folder / "score.stderr.log").write_text(redact(scored.stderr or "", env))
                if scored.returncode == 0:
                    metrics["scoring_result"] = strict_json(scored.stdout)
                    metrics["score"] = metrics["scoring_result"].get("score")
                    metrics["scoring_status"] = "SCORED — REAL RUN, COVERAGE REPORTED SEPARATELY"
                else:
                    metrics["scoring_status"] = "SCORER_FAILED"
            except KeyboardInterrupt:
                metrics["scoring_status"], operational = "SCORING_INTERRUPTED", "INTERRUPTED"
            except Exception as exc:
                metrics["scoring_status"] = "SCORER_ERROR:" + type(exc).__name__
        if operational:
            metrics["status"] = operational
        unchanged = fingerprint(data) == cohort_fingerprint
        metrics["input_code_unchanged"] = unchanged
        write_new(folder / "metrics.json", metrics)
        results[name] = {"metrics": metrics, "findings": findings["findings"] if findings and metrics["schema_valid"] else None}
        if operational or not unchanged or metrics["termination_reason"] in ("initialization_failure", "api_failure", "run_failure"):
            break
    summary = {"simulation_only": simulation, "model": env["LLM_MODEL"], "runs": results,
               "not_executed": [name for name in names if name not in results], "finding_differences": finding_differences(results)}
    summary_path = destination / ("results.json" if not (destination / "results.json").exists() else
                                  "results-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    write_new(summary_path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--only", nargs="+", choices=RUNS, default=list(RUNS))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--allow-real", action="store_true")
    parser.add_argument("--max-calls", type=int, default=48)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--max-tools", type=int, default=400)
    parser.add_argument("--max-context-chars", type=int, default=100000)
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--max-cost-usd", type=str, default="0.25")
    parser.add_argument('--require-cost-metadata', action='store_true')
    parser.add_argument('--investigation-mode', choices=('legacy', 'scheduled'), default='scheduled')
    args = parser.parse_args(argv)
    if not args.execute:
        print(exact_json({"execution_enabled": False, "runs": [{"name": n, "retrieval_mode": RUNS[n], "status": NOT_RUN,
                           "score": None, "cost_usd": None, "tokens": None, "duration": None} for n in args.only]}))
        return 0
    try:
        env = environment_for_execution(args.simulation, args.allow_real)
        if args.root.resolve().is_relative_to(args.data.resolve()):
            raise ValueError("Experiment outputs must be outside the supplied input data")
        if len(set(args.only)) != len(args.only) or any(getattr(args, n) <= 0 for n in
            ("max_calls", "max_seconds", "max_tools", "max_context_chars", "max_output_tokens")):
            raise ValueError("Run names must be unique and budgets positive")
        cost = Decimal(args.max_cost_usd)
        if not cost.is_finite() or cost <= 0:
            raise ValueError("Cost budget must be finite and positive")
        budgets = [v for key in ("max_calls", "max_seconds", "max_tools", "max_context_chars", "max_output_tokens", "max_cost_usd")
                   for v in ("--" + key.replace("_", "-"), str(getattr(args, key)))]
        budgets += ['--investigation-mode', args.investigation_mode]
        if args.require_cost_metadata:
            budgets.append('--require-cost-metadata')
        summary = execute_runs(args.data.resolve(), args.root.resolve(), args.only, env, args.simulation, budgets)
        print(exact_json({"runs_attempted": len(summary["runs"]), "simulation_only": args.simulation,
                          "remaining": summary["not_executed"]}))
        return 0
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(2, "Experiment runner refused: " + type(exc).__name__ + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
