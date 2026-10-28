"""Controller failures observed in the first real cohort; offline regressions."""
import json
import unittest

from test_phase1 import ScriptLLM, payload, call
import test_phase3
from tds_agent.agent import Agent, Limits


class RealCohortRegressionTests(unittest.TestCase):
    setup_case = test_phase3.InvestigationTests.setup_case

    def test_oversized_tool_batch_is_refused_with_matching_ids_and_recoverable(self):
        _, _, out, data, _, dispatch = self.setup_case()
        calls = [call("get_party_summary", ident=f"oversized_{i}") for i in range(17)]
        agent = Agent(ScriptLLM([payload(calls=calls), payload('{"findings":[]}')]), dispatch, out, Limits(max_calls=2))
        self.assertEqual(agent.run(), "model_call_limit")
        results = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in results], [c["id"] for c in calls])
        self.assertTrue(all(json.loads(m["content"])["error_code"] == "RUN_LIMIT" for m in results))
        self.assertEqual(data.coverage["V1"], "unprocessed")

    def test_newest_result_batch_is_bounded_before_context_delivery(self):
        documents = {f"doc{i}": "Vendor payment agreement.\n" + "X" * 5900 for i in range(6)}
        _, _, out, data, financial, dispatch = self.setup_case(documents=documents)
        calls = [call("read_document", json.dumps({"doc_id": doc}), ident=f"read_{i}") for i, doc in enumerate(documents)]
        agent = Agent(ScriptLLM([payload(calls=calls), payload('{"findings":[]}')]), dispatch, out, Limits(max_calls=2))
        self.assertEqual(agent.run(), "model_call_limit")
        results = [m for m in agent.messages if m["role"] == "tool"]
        self.assertLess(sum(len(m["content"]) for m in results), 100000 // 3)
        self.assertEqual(json.loads(results[-1]["content"])["error_code"], "RESULT_TOO_LARGE")
        self.assertFalse(financial.investigation.document_complete("doc5"))
        self.assertEqual(data.coverage["V1"], "unprocessed")

    def test_repeated_unchanged_retrieval_gets_error_without_aborting_assessments(self):
        _, _, out, data, _, dispatch = self.setup_case()
        replies = [payload(calls=[call("get_party_summary", ident=f"repeat{i}")]) for i in range(3)]
        agent = Agent(ScriptLLM(replies + [payload('{"findings":[]}')]), dispatch, out, Limits(max_calls=4))
        self.assertEqual(agent.run(), "model_call_limit")
        results = [json.loads(m["content"]) for m in agent.messages if m["role"] == "tool"]
        self.assertFalse(results[2]["success"])
        self.assertEqual(agent.attempts, 4)
        self.assertFalse(data.coverage_report()["complete"])
