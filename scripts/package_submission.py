#!/usr/bin/env python3
"""Package allowlisted submission files; retain required starter assets opaquely.

README says to zip this folder. Original encoded reviewer/key and codec files
are required by the unmodified starter/scorer. Only the archiver handles their
bytes; this script never reads, decodes or content-scans those opaque originals.
"""
import argparse
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("tds_agent", "toolkit", "data", "tests", "experiments", "examples", "scripts", "out")
ROOT_FILES = ("run.py", "requirements.txt", "findings.schema.json", "score.py", "README.md", "PRIMER.md", "DATA.md",
              "NOTES.md", "SUBMISSION_AUDIT.md",
              "answer_key_dev.enc")
EXCLUDED = {".venv", ".git", "__pycache__", ".pytest_cache", ".DS_Store"}


def opaque(path):
    return ".reviewer" in path.parts or path.name == "answer_key_dev.enc" or path.as_posix() == "toolkit/_codec.py"


def selected_files():
    paths = {Path(name) for name in ROOT_FILES if (ROOT / name).is_file()}
    for directory in DIRECTORIES:
        for current, directories, filenames in os.walk(ROOT / directory, followlinks=False):
            directories[:] = [name for name in directories if name not in EXCLUDED and not name.startswith(".env")]
            for name in filenames:
                path = (Path(current) / name).relative_to(ROOT)
                if any(p in EXCLUDED or p.startswith(".env") for p in path.parts) or name.endswith((".tmp", ".pyc", ".pyo")):
                    continue
                if directory == "experiments":
                    if len(path.parts) == 2:
                        if path.suffix not in (".py", ".md", ".json"):
                            continue
                    else:
                        # Preserve measured real cohorts, including unsuccessful
                        # runs. Never package local mock/simulation cohorts.
                        parent = ROOT / path.parent
                        metrics_path = parent / "metrics.json"
                        accepted = False
                        if metrics_path.is_file():
                            metadata = json.loads(metrics_path.read_text())
                            accepted = metadata.get("simulation_only") is False
                        elif name.startswith("results") and path.suffix == ".json":
                            metadata = json.loads((ROOT / path).read_text())
                            accepted = metadata.get("simulation_only") is False
                        if not accepted:
                            continue
                paths.add(path)
    for path in paths:
        if (ROOT / path).is_symlink():
            raise ValueError("Refusing symlink in submission manifest")
    return sorted(paths)


def secret_scan(paths):
    patterns = {"api_key_pattern": r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{16,}",
                "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                "bearer_token": r"Bearer\s+[A-Za-z0-9._-]{16,}",
                "aws_access_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
                "github_token": r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"}
    configured = [v for k, v in os.environ.items() if len(v) >= 8 and any(term in k.upper() for term in
                  ("API_KEY", "ACCESS_TOKEN", "PASSWORD", "CLIENT_SECRET"))]
    matches, scanned, skipped = [], [], []
    for path in paths:
        if opaque(path):
            skipped.append(str(path))
            continue
        text = (ROOT / path).read_bytes().decode("utf-8", errors="replace")
        scanned.append(str(path))
        for name, pattern in patterns.items():
            if re.search(pattern, text):
                matches.append({"file": str(path), "type": name})
        if any(value in text for value in configured):
            matches.append({"file": str(path), "type": "configured_environment_secret"})
    return {"scanned_files": scanned, "opaque_uninspected_files": skipped, "matches": matches,
            "scope": "All selected public files; opaque original protected/codec assets intentionally not content-inspected"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "submission/tds-take-home-offline.zip")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if archive.exists():
        raise FileExistsError("Refusing to overwrite an existing archive")
    required = ("run.py", "NOTES.md", "SUBMISSION_AUDIT.md", "out/findings.json", "out/trace.jsonl",
                "toolkit/_codec.py", "data/.reviewer/facts.enc", "answer_key_dev.enc")
    if any(not (ROOT / name).is_file() for name in required):
        raise FileNotFoundError("Required submission/runtime starter asset missing (contents not inspected)")
    paths = selected_files()
    paths.append(Path("SUBMISSION_MANIFEST.txt"))
    paths = sorted(set(paths))
    manifest = "\n".join(ROOT.name + "/" + path.as_posix() for path in paths) + "\n"
    (ROOT / "SUBMISSION_MANIFEST.txt").write_text(manifest)
    scan = secret_scan(paths)
    if scan["matches"]:
        raise ValueError("Secret-pattern scan failed; inspect filenames/types without printing secret values: " + json.dumps(scan["matches"]))
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["zip", "-q", "-X", str(archive), *[ROOT.name + "/" + path.as_posix() for path in paths]],
                   cwd=ROOT.parent, check=True)
    subprocess.run(["unzip", "-tq", str(archive)], check=True)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if names != manifest.splitlines() or len(names) != len(set(names)):
            raise ValueError("Archive manifest mismatch")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe archive path")
    output = json.loads((ROOT / "out/findings.json").read_text())
    real_metrics = [json.loads(p.read_text()) for p in (ROOT / "experiments").rglob("metrics.json")]
    real_metrics = [m for m in real_metrics if m.get("simulation_only") is False]
    report = {"archive": str(archive), "members": len(paths), "manifest_verified": True,
              "crc_verified": True, "secret_scan": scan,
              "selected_output_reported_coverage_complete": output.get("run", {}).get("coverage_complete", False),
              "real_runs_recorded": len(real_metrics), "real_runs_scored": sum(m.get("score") is not None for m in real_metrics),
              "note": "Archive integrity and output coverage do not establish financial correctness; see measured evaluation."}
    report_path = archive.with_suffix(".audit.json")
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"archive": str(archive), "members": len(paths), "secret_matches": 0,
                      "opaque_uninspected_members": len(scan["opaque_uninspected_files"]),
                      "report": str(report_path)}))


if __name__ == "__main__":
    main()
