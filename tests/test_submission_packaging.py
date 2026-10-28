"""Final archive selection preserves real attempts and excludes local secrets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import package_submission as package


class PackagingTests(unittest.TestCase):
    def test_submission_keeps_deliverable_docs_but_omits_internal_planning_docs(self):
        self.assertIn("NOTES.md", package.ROOT_FILES)
        self.assertIn("SUBMISSION_AUDIT.md", package.ROOT_FILES)
        for name in ("AGENTS.md", "BUILD_PLAN.md", "PHASE1.md", "PHASE2.md", "PHASE3.md", "PHASE5.md"):
            self.assertNotIn(name, package.ROOT_FILES)

    def test_real_cohorts_included_but_simulation_and_environment_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder, simulated in (("experiments/baseline_1", False),
                                      ("experiments/recovery/variant_1", False),
                                      ("experiments/mock/baseline_1", True)):
                target = root / folder
                target.mkdir(parents=True)
                (target / "metrics.json").write_text(json.dumps({"simulation_only": simulated}))
                (target / "trace.jsonl").write_text('{}\n')
                (target / ".env").write_text('NOT_A_REAL_CREDENTIAL')
            with patch.object(package, "ROOT", root):
                selected = {str(p) for p in package.selected_files()}
            self.assertIn("experiments/baseline_1/trace.jsonl", selected)
            self.assertIn("experiments/recovery/variant_1/trace.jsonl", selected)
            self.assertFalse(any("mock" in p or ".env" in p for p in selected))

    def test_configured_secret_detected_without_printing_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "public.txt").write_text("synthetic-secret-canary")
            with patch.object(package, "ROOT", root), patch.dict(package.os.environ,
                                                                  {"LLM_API_KEY": "synthetic-secret-canary"}):
                result = package.secret_scan([Path("public.txt")])
            self.assertEqual(result["matches"], [{"file": "public.txt", "type": "configured_environment_secret"}])
            self.assertNotIn("synthetic-secret-canary", json.dumps(result))
