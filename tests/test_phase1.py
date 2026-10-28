"""Run with python -m unittest discover -s tests -v. No real model calls."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tds_agent.agent import Agent, Limits
from tds_agent.output import INCOMPLETE, Output, strict_json

ROOT = Path(__file__).resolve().parents[1]
HAS_SCHEMA = importlib.util.find_spec("jsonschema") is not None
HAS_OPENAI = importlib.util.find_spec("openai") is not None


def payload(content=None, calls=None, finish=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.0}}


def call(name="list_tables", arguments="{}", ident="call_1"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": arguments}}


class ScriptLLM:
    model = "scripted-unit-test"

    def __init__(self, steps):
        self.steps, self.requests = list(steps), []

    def chat(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        step = self.steps[min(len(self.requests) - 1, len(self.steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return SimpleNamespace(model_dump=lambda **_: copy.deepcopy(step))


class OutputTests(unittest.TestCase):
    def test_initial_output_before_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Output(tmp)
            state = json.loads((Path(tmp) / "findings.json").read_text())
            self.assertEqual(state["findings"], [])
            self.assertIn(INCOMPLETE, state["limitations"])
            self.assertIsNone(out.validator)
            self.assertEqual(json.loads(out.trace_path.read_text())["event"], "run_started")

    def test_trace_appends_with_run_identity_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"LLM_API_KEY": "private-test-key"}):
            out = Output(tmp)
            out.trace("test", content="private-test-key", api_key="different-secret")
            out2 = Output(tmp)
            records = [json.loads(line) for line in out.trace_path.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertNotEqual(out.run_id, out2.run_id)
            self.assertNotIn("private-test-key", out.trace_path.read_text())
            self.assertNotIn("different-secret", out.trace_path.read_text())
            self.assertTrue(all(r["ts"] for r in records))

    def test_strict_json_rejects_duplicates_and_nonfinite(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json(text)

    def test_atomic_write_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Output(tmp)
            before = (Path(tmp) / "findings.json").read_bytes()
            with patch("tds_agent.output.os.replace", side_effect=OSError("injected disk failure")):
                with self.assertRaises(OSError):
                    out._write({"findings": []})
            self.assertEqual(before, (Path(tmp) / "findings.json").read_bytes())
            self.assertFalse(list(Path(tmp).glob(".findings-*.tmp")))

    def test_dependency_initialization_failure_still_has_outputs(self):
        import run

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Output, "attach_schema", side_effect=ImportError("injected missing dependency")):
                code = run.main(["--data", "missing-input", "--out", tmp])
            self.assertEqual(code, 1)
            state = json.loads((Path(tmp) / "findings.json").read_text())
            self.assertEqual(state["run"]["termination_reason"], "initialization_failure")
            self.assertEqual(state["findings"], [])

    def test_interruption_finalizes(self):
        import run

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Output, "attach_schema", side_effect=KeyboardInterrupt()):
                self.assertEqual(run.main(["--out", tmp]), 130)
            state = json.loads((Path(tmp) / "findings.json").read_text())
            self.assertEqual(state["run"]["termination_reason"], "interrupted")


class LoopTests(unittest.TestCase):
    """Unit tests isolate provider and dispatcher; schema/Toolkit tested below."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Output(self.tmp.name)
        self.out.validator = Mock()  # Only loop sequencing, not schema validation.
        self.dispatcher = SimpleNamespace(specs=[{"type": "function", "function": {"name": "test"}}],
                                          execute=Mock(return_value='{"success":true,"data":{}}'))

    def agent(self, steps, **limits):
        llm = ScriptLLM(steps)
        return Agent(llm, self.dispatcher, self.out, Limits(**limits), sleep=lambda _: None)

    def test_multiple_tools_have_matched_results_before_next_model_call(self):
        agent = self.agent([payload(calls=[call(ident="a"), call(ident="b")]), payload('{"findings":[]}')])
        self.assertEqual(agent.run(), "phase1_finished")
        request = agent.llm.requests[1]
        self.assertEqual([m["tool_call_id"] for m in request if m["role"] == "tool"], ["a", "b"])

    def test_transient_retry_logs_every_attempt_and_keeps_same_messages(self):
        agent = self.agent([TimeoutError("transient"), payload('{"findings":[]}')])
        self.assertEqual(agent.run(), "phase1_finished")
        self.assertEqual(agent.attempts, 2)
        self.assertEqual(agent.llm.requests[0], agent.llm.requests[1])
        self.assertIsNone(agent.metrics()["cost_usd"])
        events = [json.loads(line)["event"] for line in self.out.trace_path.read_text().splitlines()]
        self.assertEqual(events.count("model_attempt"), 2)
        self.assertIn("model_error", events)

    def test_retry_exhaustion_and_permanent_error(self):
        agent = self.agent([TimeoutError("persistent")], max_retries=2)
        self.assertEqual(agent.run(), "api_failure")
        self.assertEqual(agent.attempts, 3)
        agent = self.agent([ValueError("bad request")])
        self.assertEqual(agent.run(), "api_failure")
        self.assertEqual(agent.attempts, 1)

    def test_max_calls_includes_retries(self):
        agent = self.agent([TimeoutError()], max_calls=1)
        self.assertEqual(agent.run(), "model_call_limit")
        self.assertEqual(agent.attempts, 1)

    def test_repeated_actions_stop(self):
        agent = self.agent([payload(calls=[call()])])
        self.assertEqual(agent.run(), "repeated_action")
        self.assertEqual(agent.attempts, 3)
        self.assertIsNotNone(self.dispatcher.execute.call_args.kwargs["blocked"])

    def test_invalid_final_and_invented_financial_findings_are_rejected(self):
        agent = self.agent([payload("not JSON"), payload('{"findings":[{"invented":true}]}'), payload('{"findings":[]}')])
        self.assertEqual(agent.run(), "phase1_finished")
        self.assertEqual(self.out.state["findings"], [])
        self.assertIn(INCOMPLETE, self.out.state["limitations"])
        events = [json.loads(line)["event"] for line in self.out.trace_path.read_text().splitlines()]
        self.assertEqual(events.count("invalid_final"), 2)

    def test_repeated_invalid_final_and_empty_responses_are_bounded(self):
        agent = self.agent([payload("not JSON")])
        self.assertEqual(agent.run(), "repeated_response")
        agent = self.agent([{"choices": [], "usage": None}], max_calls=2)
        self.assertEqual(agent.run(), "model_call_limit")

    def test_truncated_tools_are_not_executed_and_have_results(self):
        agent = self.agent([payload(calls=[call(ident="a"), call(ident="b")], finish="length")])
        self.assertEqual(agent.run(), "truncated_response")
        self.assertTrue(all(c.kwargs["blocked"] for c in self.dispatcher.execute.call_args_list))
        self.assertEqual(len([m for m in agent.messages if m["role"] == "tool"]), 2)

    def test_context_limit_and_duplicate_ids(self):
        agent = self.agent([payload()], max_context_chars=1)
        self.assertEqual(agent.run(), "context_limit")
        self.assertEqual(agent.attempts, 0)
        agent = self.agent([payload(calls=[call(), call()])])
        self.assertEqual(agent.run(), "invalid_tool_call_ids")

    def test_time_limit(self):
        agent = self.agent([payload()])
        agent.started -= 1000
        self.assertEqual(agent.run(), "time_limit")
        self.assertEqual(agent.attempts, 0)


@unittest.skipUnless(HAS_SCHEMA, "jsonschema dependency unavailable")
class SchemaTests(unittest.TestCase):
    def test_schema_validation_and_partial_checkpoint_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Output(tmp)
            out.attach_schema(ROOT / "findings.schema.json")
            # Synthetic fixture tests output preservation, not financial accuracy.
            candidate = copy.deepcopy(out.state)
            candidate["findings"] = [{"party": "Synthetic party", "section": "unknown", "status": "needs_review",
                                      "shortfall": 0, "reason": "Synthetic unresolved fact", "needs": "Contract scope"}]
            out.checkpoint(candidate)
            with self.assertRaises(Exception):
                out.checkpoint({"findings": [{"party": "invalid"}]})
            out.finalize("api_failure")
            saved = json.loads((Path(tmp) / "findings.json").read_text())
            self.assertEqual(saved["findings"], candidate["findings"])
            out.validator.validate(saved)

    def test_dispatch_validation_errors_and_tool_exceptions(self):
        from tds_agent.tools import ToolDispatcher

        specs = [{"function": {"name": "read_document", "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
            "required": ["doc_id"], "additionalProperties": False}}}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Output(tmp)
            fn = Mock(return_value='{"success":true,"data":"test"}')
            dispatcher = ToolDispatcher(SimpleNamespace(specs=specs, functions={"read_document": fn}), out)
            for arguments in ('{"doc_id":', '[]', '{"doc_id":5}', '{"doc_id":"a","extra":1}',
                              '{"doc_id":"../private"}', '{"doc_id":"a","offset":true}',
                              '{"doc_id":"a","limit":999999}'):
                result = json.loads(dispatcher.execute("id", "read_document", arguments))
                self.assertEqual(result["error_code"], "INVALID_ARGUMENTS")
            fn.assert_not_called()
            self.assertEqual(json.loads(dispatcher.execute("id", "unknown", "{}"))["error_code"], "UNKNOWN_TOOL")
            fn.side_effect = RuntimeError("injected tool failure")
            self.assertEqual(json.loads(dispatcher.execute("id", "read_document", '{"doc_id":"a"}'))["error_code"], "TOOL_ERROR")


@unittest.skipUnless(HAS_SCHEMA and HAS_OPENAI, "openai/jsonschema dependencies unavailable")
class FakeIntegrationTests(unittest.TestCase):
    def run_script(self, script, limits=None, tool_failure=False):
        from toolkit import LLM, Toolkit
        from toolkit.fake_llm import FakeLLM
        from tds_agent.tools import ToolDispatcher

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        server = FakeLLM(script).start()
        self.addCleanup(server.stop)
        out = Output(tmp.name)
        out.attach_schema(ROOT / "findings.schema.json")
        kit = Toolkit(ROOT / "data", log_dir=tmp.name)
        if tool_failure:
            kit.functions["list_tables"] = Mock(side_effect=RuntimeError("injected tool execution failure"))
        agent = Agent(LLM(base_url=server.url, api_key="fake", model="fake-test"),
                      ToolDispatcher(kit, out), out, limits, sleep=lambda _: None)
        reason = agent.run()
        out.finalize(reason, agent.metrics())
        out.validator.validate(json.loads((Path(tmp.name) / "findings.json").read_text()))
        records = [json.loads(line) for line in out.trace_path.read_text().splitlines()]
        attempts = [r for r in records if r["event"] == "model_attempt"]
        terminals = [r for r in records if r["event"] in ("model_response", "model_error")]
        self.assertEqual([r["attempt"] for r in attempts], [r["attempt"] for r in terminals])
        for r in attempts:
            self.assertIn("messages", r)
            self.assertIn("tools", r)
            pending = []
            for message in r["messages"]:
                if message["role"] == "assistant":
                    self.assertFalse(pending)
                    pending = [c["id"] for c in message.get("tool_calls") or []]
                elif message["role"] == "tool":
                    self.assertIn(message["tool_call_id"], pending)
                    pending.remove(message["tool_call_id"])
            self.assertFalse(pending)
        starts = [r for r in records if r["event"] == "tool_start"]
        results = [r for r in records if r["event"] == "tool_result"]
        self.assertEqual([r["tool_call_id"] for r in starts], [r["tool_call_id"] for r in results])
        self.assertTrue(all("arguments" in r and "result" in r for r in results))
        self.assertTrue(all(r["ts"] and r["run_id"] == out.run_id for r in records))
        return agent, out, server, reason

    def test_supplied_example_script(self):
        script = json.loads((ROOT / "toolkit/scripts/example.json").read_text())
        agent, out, server, reason = self.run_script(script)
        self.assertEqual(reason, "repeated_response")  # Example ends in prose, not valid final JSON.
        records = [json.loads(line) for line in out.trace_path.read_text().splitlines()]
        codes = [r["result"].get("error_code") for r in records if r["event"] == "tool_result"]
        self.assertIn("INVALID_ARGUMENTS", codes)
        self.assertIn("NOT_FOUND", codes)
        self.assertEqual(agent.tool_calls, 4)
        self.assertEqual(agent.attempts, 7)

    def test_native_loop_valid_final_and_multiple_calls(self):
        script = {"steps": [{"tool_calls": [{"name": "list_tables"}, {"name": "list_documents"}]},
                            {"content": '{"findings":[],"limitations":["Incomplete Phase 1"]}'}]}
        agent, out, server, reason = self.run_script(script)
        self.assertEqual(reason, "phase1_finished")
        self.assertEqual(agent.attempts, 2)
        self.assertEqual(len(server.requests[1]["tool_results"]), 2)

    def test_fake_429_retry(self):
        script = {"steps": [{"content": '{"findings":[]}'}], "faults": [{"step": 0, "status": 429}]}
        agent, out, server, reason = self.run_script(script)
        self.assertEqual(reason, "phase1_finished")
        self.assertEqual(agent.attempts, 2)

    def test_malformed_unknown_missing_document_then_valid_recovery(self):
        script = {"steps": [
            {"tool_calls": [{"name": "read_document", "arguments": '{"doc_id":'}]},
            {"tool_calls": [{"name": "unknown_operation"}]},
            {"tool_calls": [{"name": "read_document", "arguments": '{"doc_id":"doc_99"}'}]},
            {"tool_calls": [{"name": "read_document", "arguments": '{"doc_id":"doc_03"}'}]},
            {"content": '{"findings":[]}'}]}
        agent, out, server, reason = self.run_script(script)
        self.assertEqual(reason, "phase1_finished")
        records = [json.loads(line) for line in out.trace_path.read_text().splitlines()]
        results = [r["result"] for r in records if r["event"] == "tool_result"]
        self.assertEqual([r.get("error_code") for r in results[:3]],
                         ["INVALID_ARGUMENTS", "UNKNOWN_TOOL", "NOT_FOUND"])
        self.assertTrue(results[3]["success"])
        self.assertIn("text", results[3]["data"])

    def test_model_failure_preserves_valid_outputs(self):
        script = {"steps": [{"tool_calls": [{"name": "list_tables"}]}],
                  "faults": [{"step": 0, "status": 500, "times": 10}]}
        agent, out, server, reason = self.run_script(script)
        self.assertEqual(reason, "api_failure")
        self.assertEqual(agent.attempts, 3)
        self.assertEqual(out.state["findings"], [])
        self.assertIsNone(out.state["run"]["cost_usd"])

    def test_tool_exception_preserves_valid_outputs_and_recovers(self):
        script = {"steps": [{"tool_calls": [{"name": "list_tables"}]}, {"content": '{"findings":[]}'}]}
        agent, out, server, reason = self.run_script(script, tool_failure=True)
        self.assertEqual(reason, "phase1_finished")
        records = [json.loads(line) for line in out.trace_path.read_text().splitlines()]
        result = next(r["result"] for r in records if r["event"] == "tool_result")
        self.assertEqual(result["error_code"], "TOOL_ERROR")
        self.assertFalse(result["success"])

    def test_cli_logs_redact_environment_secrets(self):
        from toolkit.fake_llm import FakeLLM
        from jsonschema import validate

        sentinel = "phase1-synthetic-secret-canary"
        script = {"steps": [{"content": json.dumps({"findings": [], "limitations": [sentinel]})}]}
        server = FakeLLM(script).start()
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, LLM_BASE_URL=server.url, LLM_API_KEY=sentinel, LLM_MODEL="local-test")
            proc = subprocess.run([sys.executable, str(ROOT / "run.py"), "--data", str(ROOT / "data"),
                                   "--out", tmp], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            for path in Path(tmp).glob("*"):
                if path.is_file():
                    self.assertNotIn(sentinel, path.read_text())
            self.assertNotIn(sentinel, proc.stdout + proc.stderr)
            self.assertIn("[REDACTED]", (Path(tmp) / "trace.jsonl").read_text())
            validate(json.loads((Path(tmp) / "findings.json").read_text()),
                     json.loads((ROOT / "findings.schema.json").read_text()))

    def test_cli_uses_custom_paths_and_validates_output(self):
        from toolkit.fake_llm import FakeLLM
        from jsonschema import validate

        script = {"steps": [{"tool_calls": [{"name": "list_tables"}]}, {"content": '{"findings":[]}'}]}
        server = FakeLLM(script).start()
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, LLM_BASE_URL=server.url, LLM_API_KEY="fake", LLM_MODEL="fake-test")
            proc = subprocess.run([sys.executable, str(ROOT / "run.py"), "--data", str(ROOT / "data"),
                                   "--out", str(Path(tmp) / "custom")], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads((Path(tmp) / "custom/findings.json").read_text())
            validate(result, json.loads((ROOT / "findings.schema.json").read_text()))
            self.assertFalse(result["run"]["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
