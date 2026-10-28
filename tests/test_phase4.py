"""Offline finalization tests: mocks and the existing supplied fake endpoint only."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.run_experiments import (RUNS, environment_for_execution, execute_runs,
                                        finding_differences, main, measured_artifacts)
from phase2_support import fixture, row, PAN
from tds_agent.agent import Agent
from tds_agent.output import Output, exact_json
from tds_agent.tools import FinancialTools, ToolDispatcher


class ExperimentPreparationTests(unittest.TestCase):
    def test_default_is_plan_only_and_all_five_are_unrun(self):
        capture = io.StringIO()
        with patch("subprocess.run", side_effect=AssertionError("No execution in plan mode")), contextlib.redirect_stdout(capture):
            self.assertEqual(main([]), 0)
        plan = json.loads(capture.getvalue())
        self.assertFalse(plan["execution_enabled"])
        self.assertEqual(len(plan["runs"]), 5)
        self.assertTrue(all(r["status"] == "NOT RUN — API ACCESS UNAVAILABLE" and r["cost_usd"] is None for r in plan["runs"]))

    def test_explicit_environment_localhost_and_real_execution_guards(self):
        env = {"LLM_BASE_URL": "http://127.0.0.1:1234/v1", "LLM_MODEL": "local", "LLM_API_KEY": "fake"}
        self.assertEqual(environment_for_execution(True, False, env)["NO_PROXY"], "localhost,127.0.0.1,::1")
        for bad, simulation, allow in (({}, True, False), ({**env, "LLM_BASE_URL": "https://example.invalid/v1"}, True, False),
                                       ({**env, "LLM_BASE_URL": "https://example.invalid/v1"}, False, False),
                                       ({**env, "LLM_BASE_URL": "http://user:secret@localhost/v1"}, True, False)):
            with self.assertRaises(ValueError):
                environment_for_execution(simulation, allow, bad)

    def test_no_prior_run_is_overwritten_even_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "baseline_2").mkdir()
            existing = folder / "baseline_2/findings.json"
            existing.write_text('{"findings":[]}')
            with self.assertRaises(FileExistsError):
                execute_runs(Path("data").resolve(), folder, ["baseline_1", "baseline_2"], {}, True, [],
                             invoke=lambda *a, **k: self.fail("Should not start a child"))
            self.assertFalse((folder / "baseline_1").exists())
            self.assertEqual(existing.read_text(), '{"findings":[]}')

    def test_mocked_cohort_changes_only_retrieval_and_preserves_unavailable_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            calls = []
            def invoke(command, **kwargs):
                calls.append(command)
                folder = Path(command[command.index("--out") + 1])
                output = Output(folder)
                output.finalize("model_call_limit")
                return SimpleNamespace(returncode=1, stdout="fake-test-secret", stderr="")
            env = {"LLM_MODEL": "fixed-model", "LLM_BASE_URL": "http://localhost:1234/v1", "LLM_API_KEY": "fake-test-secret"}
            result = execute_runs(Path("data").resolve(), destination, list(RUNS), env, True, ["--max-calls", "1"], invoke)
            self.assertEqual(len(calls), 5)
            self.assertEqual([c[c.index("--retrieval-mode") + 1] for c in calls], ["agent"] * 3 + ["proactive"] * 2)
            for name in RUNS:
                metrics = result["runs"][name]["metrics"]
                self.assertIsNone(metrics["cost_usd"])
                self.assertIsNone(metrics["score"])
                self.assertEqual(metrics["status"], "INCOMPLETE")
                self.assertIn("[REDACTED]", (destination / name / "stdout.log").read_text())
                self.assertTrue((destination / name / "trace.jsonl").exists())

    def test_interruption_records_attempt_and_leaves_other_runs_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def invoke(command, **kwargs):
                Output(command[command.index("--out") + 1]).finalize("interrupted")
                raise KeyboardInterrupt()
            env = {"LLM_MODEL": "local", "LLM_BASE_URL": "http://localhost:1234/v1", "LLM_API_KEY": "fake"}
            summary = execute_runs(Path("data").resolve(), Path(tmp), list(RUNS), env, True, [], invoke)
            self.assertEqual(summary["runs"]["baseline_1"]["metrics"]["status"], "INTERRUPTED")
            self.assertEqual(len(summary["not_executed"]), 4)

    def test_differences_use_actual_findings_and_preserve_duplicate_counts(self):
        a = {"party": "Synthetic", "section": "194C", "shortfall": 1}
        differences = finding_differences({"a": {"findings": [a, a]}, "b": {"findings": [a]}})
        self.assertEqual(differences[0]["only_left"], [a])
        self.assertEqual(differences[0]["only_right"], [])

    def test_proactive_variant_uses_full_checked_sources_and_explicit_omission(self):
        _, kit, output, data = fixture(self, documents={"agreement": f"Vendor PAN {PAN}: machinery is supplied on dry hire."})
        financial = FinancialTools(data, output, investigate=True)
        inv = financial.investigation
        empty = inv.proactive_context(max_chars=1)
        self.assertEqual(empty["omitted_document_ids"], ["agreement"])
        self.assertFalse(inv.document_complete("agreement"))
        populated = inv.proactive_context()
        self.assertEqual(len(populated["documents"]), 1)
        self.assertTrue(inv.document_complete("agreement"))
        self.assertEqual(data.coverage["V1"], "unprocessed")

    def test_both_modes_execute_existing_cli_on_fake_without_scoring(self):
        from toolkit.fake_llm import FakeLLM
        server = FakeLLM({"steps": [{"content": '{"findings":[]}'}]}).start()
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            env = environment_for_execution(True, False, {**dict(__import__('os').environ), "LLM_BASE_URL": server.url,
                                                            "LLM_MODEL": "fixed-local-model", "LLM_API_KEY": "fake"})
            result = execute_runs(Path("data").resolve(), Path(tmp), ["baseline_1", "variant_1"], env, True,
                                  ["--max-calls", "1"])
            self.assertTrue(result["simulation_only"])
            for run in result["runs"].values():
                self.assertEqual(run["metrics"]["model_attempts"], 1)
                self.assertEqual(run["metrics"]["cost_usd"], 0)
                self.assertEqual(run["metrics"]["scoring_status"], "NOT SCORED — SIMULATION ONLY")
                self.assertFalse(run["metrics"]["coverage_complete"])
