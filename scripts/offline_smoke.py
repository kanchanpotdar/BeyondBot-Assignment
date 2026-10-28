#!/usr/bin/env python3
"""Extract an archive and verify its own code using only a local supplied fake."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    report_path = archive.with_suffix(".smoke.json")
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite an earlier smoke report")
    original_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="tds-submission-smoke-") as temporary:
            subprocess.run(["unzip", "-q", str(archive), "-d", temporary], check=True)
            project = (Path(temporary) / "tds-take-home").resolve()
            originals = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
                         for name in ("out/findings.json", "out/trace.jsonl")}
            os.chdir(project)
            sys.path.insert(0, str(project))
            import toolkit
            from toolkit.fake_llm import FakeLLM

            assert Path(toolkit.__file__).resolve().is_relative_to(project)
            server = FakeLLM(json.loads((project / "toolkit/scripts/example.json").read_text())).start()
            try:
                env = dict(os.environ, LLM_BASE_URL=server.url, LLM_API_KEY="offline-extracted-smoke-dummy",
                           LLM_MODEL="offline-fake-extracted", NO_PROXY="localhost,127.0.0.1,::1")
                for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                            "http_proxy", "https_proxy", "all_proxy", "PHASE3_SAVE_ARTIFACTS"):
                    env.pop(key, None)
                help_result = subprocess.run([sys.executable, "run.py", "--help"], env=env,
                                             capture_output=True, text=True, timeout=10)
                assert help_result.returncode == 0 and "--data" in help_result.stdout and "--out" in help_result.stdout
                run = subprocess.run([sys.executable, "-B", "run.py", "--data", "data", "--out", "smoke_out", "--max-calls", "8"],
                                     env=env, capture_output=True, text=True, timeout=30)
                assert run.returncode == 1 and "repeated_response" in run.stdout
                audit = subprocess.run([sys.executable, "-B", "tests/verify_phase3_outputs.py", "--out", "smoke_out"],
                                       env=env, capture_output=True, text=True, timeout=10)
                assert audit.returncode == 0, "Extracted output schema/trace audit failed"
                suite = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-q"],
                                       env=env, capture_output=True, text=True, timeout=90)
                total = re.search(r"Ran (\d+) tests", suite.stderr)
                skipped = re.search(r"skipped=(\d+)", suite.stderr)
                skipped_count = int(skipped[1]) if skipped else 0
                assert suite.returncode == 0 and total and "OK" in suite.stderr and skipped_count == 0, "Extracted regression suite failed or skipped tests"
                assert all(hashlib.sha256((project / name).read_bytes()).hexdigest() == value for name, value in originals.items())
                report = {"archive": str(archive), "extracted_cli_help_valid": True, "smoke_exit_code": run.returncode,
                          "expected_termination": "repeated_response", "smoke_output_audit": json.loads(audit.stdout),
                          "extracted_tests": {"passed": int(total[1]) - skipped_count, "failed": 0, "skipped": skipped_count},
                          "archived_out_preserved": True, "real_api_calls": 0,
                          "runtime": "Existing isolated .venv; no downloads or system Python modifications"}
            finally:
                server.stop()
            os.chdir(original_cwd)
    finally:
        os.chdir(original_cwd)
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
