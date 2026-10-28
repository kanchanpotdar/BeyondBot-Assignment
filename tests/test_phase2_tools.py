import copy
import json
import unittest
from decimal import Decimal
from unittest.mock import patch

from phase2_support import fixture, row, party, cert, turnover, PAN, OTHER_PAN
from tds_agent.agent import Agent, Limits
from tds_agent.output import strict_json
from tds_agent.tools import FinancialTools, ToolDispatcher


def classification(vouchers=None, section="194C", category="contractor", **extra):
    return {"vouchers": vouchers or ["V1"], "section": section, "category": category,
            "evidence": (vouchers or ["V1"])[:1], **extra}


class FinancialToolTests(unittest.TestCase):
    def setup_tools(self, **kwargs):
        root, kit, out, data = fixture(self, **kwargs)
        tools = FinancialTools(data, out)
        dispatch = ToolDispatcher(kit, out, financial=tools)
        return root, kit, out, data, tools, dispatch

    def test_exact_names_and_strict_arguments(self):
        _, _, _, data, tools, dispatch = self.setup_tools()
        self.assertEqual(set(tools.functions), {"get_party_summary", "get_party_transactions", "calculate_tds", "record_assessment"})
        requests = [
            ("get_party_summary", {"payee_id": "missing"}),
            ("get_party_transactions", {"payee_id": "PAN:" + PAN, "offset": True}),
            ("calculate_tds", {"payee_id": "PAN:" + PAN, "classifications": [classification(amount="1")]}),
            ("calculate_tds", {"payee_id": "PAN:" + PAN, "classifications": [classification(rate="0")]}),
            ("calculate_tds", {"payee_id": "PAN:" + PAN, "classifications": [classification(["unknown"])]}),
            ("record_assessment", {"calculation_id": "invented", "reason": "Synthetic explanation"}),
            ("record_assessment", {"calculation_id": "invented", "reason": "Synthetic explanation", "shortfall": 0})]
        for name, args in requests:
            result = strict_json(dispatch.execute("call", name, json.dumps(args)))
            self.assertFalse(result["success"], (name, args))
            self.assertEqual(result["error_code"], "INVALID_ARGUMENTS")
        self.assertEqual(data.coverage["V1"], "unprocessed")

    def test_complete_plan_required_no_partial_thresholds(self):
        _, _, _, data, tools, _ = self.setup_tools(rows=[row("A", "60000"), row("B", "60000")])
        for plan in ([classification(["A"])], [classification(["A", "B"]), classification(["B"]) ]):
            with self.assertRaises(ValueError):
                tools.calculate_tds("PAN:" + PAN, plan)
        result = tools.calculate_tds("PAN:" + PAN, [classification(["A"]), classification(["B"])])
        self.assertEqual(len(result["calculations"]), 1)
        self.assertEqual(result["calculations"][0]["tds_required"], Decimal("2400"))
        self.assertEqual(set(data.coverage.values()), {"unprocessed"})

    def test_pan_aliases_aggregate_and_categories_remain_separate(self):
        rows = [row("A", "30000", party="First", ledger="Professional"),
                row("B", "30000", party="Alias", ledger="Professional"),
                row("C", "50000", party="Alias", ledger="Technical")]
        _, _, out, data, tools, _ = self.setup_tools(rows=rows, parties=[party("First"), party("Alias")])
        plan = [classification(["A", "B"], "194J", "professional"), classification(["C"], "194J", "technical")]
        results = tools.calculate_tds("PAN:" + PAN, plan)["calculations"]
        self.assertEqual([c["tds_required"] for c in results], [Decimal("6000"), Decimal("0")])
        for result in results:
            tools.record_assessment(result["calculation_id"], "Synthetic fees are supported by the selected source vouchers.")
        self.assertTrue(data.coverage_report()["complete"])
        self.assertEqual(len(out.state["findings"]), 1)
        self.assertEqual(out.state["findings"][0]["vouchers"], ["A", "B"])

    def test_mixed_rent_category_monthly_aggregation(self):
        rows = [row("A", "35000", ledger="Building"), row("B", "20000", ledger="Machine")]
        _, _, _, _, tools, _ = self.setup_tools(rows=rows)
        plan = [classification(["A"], "194I", "land_building_furniture"), classification(["B"], "194I", "plant_machinery_equipment")]
        results = tools.calculate_tds("PAN:" + PAN, plan)["calculations"]
        self.assertEqual([c["tds_required"] for c in results], [Decimal("3500"), Decimal("400")])

    def test_shared_rent_certificate_cap_is_not_reused_per_category(self):
        rows = [row("A", "35000", ledger="Building"), row("B", "20000", ledger="Machine")]
        _, _, _, _, tools, _ = self.setup_tools(rows=rows, documents={"certificate": cert(section="194I — rent", cap="40000", start="01-04-2025")})
        plan = [classification(["A"], "194I", "land_building_furniture", certificate_ids=["certificate"]),
                classification(["B"], "194I", "plant_machinery_equipment")]
        results = tools.calculate_tds("PAN:" + PAN, plan)["calculations"]
        self.assertEqual([c["tds_required"] for c in results], [Decimal("350"), Decimal("350")])

    def test_missing_evidence_errors_do_not_mark_compliant(self):
        _, _, out, data, tools, dispatch = self.setup_tools()
        args = {"payee_id": "PAN:" + PAN, "classifications": [classification(evidence=["does-not-exist"])]}
        result = strict_json(dispatch.execute("c", "calculate_tds", json.dumps(args)))
        self.assertFalse(result["success"])
        self.assertEqual(data.coverage["V1"], "unprocessed")
        self.assertFalse(out.state["run"]["coverage_complete"])

    def test_assessment_checkpoints_deduplicates_and_excludes_compliant(self):
        _, _, out, data, tools, _ = self.setup_tools()
        calculation = tools.calculate_tds("PAN:" + PAN, [classification()])["calculations"][0]
        for _ in range(2):
            tools.record_assessment(calculation["calculation_id"], "Synthetic contractor voucher establishes taxable work.")
        self.assertEqual(len(out.state["findings"]), 1)
        self.assertEqual(out.state["findings"][0]["shortfall"], Decimal("800"))
        self.assertTrue(data.coverage_report()["complete"])
        saved = json.loads((out.directory / "findings.json").read_text())
        out.validator.validate(saved)
        self.assertEqual(saved["findings"][0]["tds_required"], 800)

    def test_compliant_empty_findings_requires_all_recorded_coverage(self):
        _, _, out, data, tools, _ = self.setup_tools(rows=[row(amount="30000")])
        calculation = tools.calculate_tds("PAN:" + PAN, [classification()])["calculations"][0]
        self.assertFalse(out.state["run"]["coverage_complete"])
        tools.record_assessment(calculation["calculation_id"], "The single voucher equals but does not exceed the threshold.")
        out.finalize("financial_review_complete")
        self.assertEqual(out.state["findings"], [])
        self.assertTrue(out.state["run"]["coverage_complete"])

    def test_unresolved_is_financial_with_zero_shortfall_and_incomplete_coverage(self):
        _, _, out, data, tools, _ = self.setup_tools(rows=[row(amount="5100000")])
        result = tools.calculate_tds("PAN:" + PAN, [classification(section="194Q", category="goods")])["calculations"][0]
        self.assertEqual(result["status"], "needs_review")
        tools.record_assessment(result["calculation_id"], "Prior-year buyer turnover is absent from the available evidence.")
        finding = out.state["findings"][0]
        self.assertEqual(finding["shortfall"], 0)
        self.assertIn("turnover", finding["needs"])
        self.assertNotIn("tds_required", finding)
        self.assertEqual(data.coverage["V1"], "unresolved")
        self.assertFalse(out.state["run"]["coverage_complete"])

    def test_checkpoint_failure_preserves_prior_results_and_coverage(self):
        _, _, out, data, tools, _ = self.setup_tools()
        result = tools.calculate_tds("PAN:" + PAN, [classification()])["calculations"][0]
        prior = (out.directory / "findings.json").read_bytes()
        with patch.object(out, "_write", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                tools.record_assessment(result["calculation_id"], "Supported synthetic contractor explanation.")
        self.assertEqual((out.directory / "findings.json").read_bytes(), prior)
        self.assertEqual(data.coverage["V1"], "unprocessed")

    def test_new_plan_invalidates_old_unrecorded_ids_and_recorded_plan_cannot_change(self):
        _, _, _, _, tools, _ = self.setup_tools()
        old = tools.calculate_tds("PAN:" + PAN, [classification()])["calculations"][0]["calculation_id"]
        new = tools.calculate_tds("PAN:" + PAN, [classification(section="194J", category="director")])["calculations"][0]["calculation_id"]
        with self.assertRaises(ValueError):
            tools.record_assessment(old, "An old stale classification must not be accepted.")
        tools.record_assessment(new, "Synthetic source establishes director sitting fees.")
        with self.assertRaises(ValueError):
            tools.calculate_tds("PAN:" + PAN, [classification()])

    def test_ledger_selector_cannot_invent_entities_or_skip_vouchers(self):
        _, _, _, _, tools, _ = self.setup_tools()
        plan = [{"section": "194C", "category": "contractor", "ledgers": ["Services"], "evidence": ["V1"]}]
        result = tools.calculate_tds("PAN:" + PAN, plan)
        self.assertEqual(result["calculations"][0]["tds_required"], Decimal("800"))
        plan[0]["ledgers"] = ["Corporate Group"]
        with self.assertRaises(ValueError):
            tools.calculate_tds("PAN:" + PAN, plan)

    def test_oversized_result_is_explicit_and_full_result_is_traced_once(self):
        _, kit, out, data, tools, _ = self.setup_tools(rows=[row(str(i)) for i in range(10)])
        dispatch = ToolDispatcher(kit, out, max_result_chars=500, financial=tools)
        args = json.dumps({"payee_id": "PAN:" + PAN, "limit": 10})
        result = strict_json(dispatch.execute("size-test", "get_party_transactions", args))
        self.assertEqual(result["error_code"], "RESULT_TOO_LARGE")
        records = [strict_json(line) for line in out.trace_path.read_text().splitlines()]
        raw = [r for r in records if r["event"] == "tool_result" and r["tool_call_id"] == "size-test"]
        self.assertEqual(len(raw), 1)
        self.assertEqual(len(raw[0]["result"]["data"]["items"]), 10)
        self.assertEqual(set(data.coverage.values()), {"unprocessed"})

    def test_api_failure_after_record_cannot_discard_prior_findings(self):
        # Last validated assessments, rather than any model final JSON, remain
        # the financial source of truth during output finalization.
        _, _, out, data, tools, _ = self.setup_tools()
        result = tools.calculate_tds("PAN:" + PAN, [classification()])["calculations"][0]
        tools.record_assessment(result["calculation_id"], "Synthetic payment exceeds the single contractor threshold.")
        before = copy.deepcopy(out.state["findings"])
        with patch.object(out, "validator") as validator:
            validator.validate.side_effect = ValueError("injected invalid checkpoint")
            with self.assertRaises(ValueError):
                out.checkpoint({"findings": []})
        out.finalize("api_failure")
        self.assertEqual(out.state["findings"], before)

    def test_excluded_assessment_retained_in_internal_coverage(self):
        _, _, out, data, tools, _ = self.setup_tools()
        result = tools.calculate_tds("PAN:" + PAN, [classification(section="excluded", category="excluded")])["calculations"][0]
        tools.record_assessment(result["calculation_id"], "Synthetic evidence establishes an employee reimbursement at actuals.")
        self.assertEqual(out.state["findings"], [])
        self.assertEqual(data.coverage["V1"], "compliant")
        self.assertTrue(data.assessments)

    def test_synthetic_reviewer_fact_is_bound_to_actual_tool_response_id(self):
        _, kit, _, data, tools, dispatch = self.setup_tools(rows=[row(amount="5100000")])
        # Explicit synthetic test double, not a fabricated client interaction.
        kit.reviewer.ask.return_value = {"answer": turnover(), "questions_left": 2}
        response = strict_json(dispatch.execute("review-call", "ask_reviewer", json.dumps({"question": "Synthetic prior-year turnover request"})))
        evidence_id = response["data"]["evidence_id"]
        self.assertEqual(evidence_id, "reviewer:review-call")
        calculation = tools.calculate_tds("PAN:" + PAN, [classification(section="194Q", category="goods", turnover_id=evidence_id)])["calculations"][0]
        self.assertEqual(calculation["tds_required"], Decimal("100"))
        self.assertIn(evidence_id, calculation["evidence"])
        kit.reviewer.ask.assert_called_once()

    def test_calculation_exception_is_structured_and_does_not_advance_coverage(self):
        _, _, out, data, tools, dispatch = self.setup_tools()
        with patch.object(tools.calculator, "calculate", side_effect=RuntimeError("injected financial tool failure")):
            response = strict_json(dispatch.execute("failed-calculation", "calculate_tds", json.dumps({"payee_id": "PAN:" + PAN, "classifications": [classification()]})))
        self.assertEqual(response["error_code"], "TOOL_ERROR")
        self.assertFalse(tools.calculations)
        self.assertEqual(data.coverage["V1"], "unprocessed")
        self.assertEqual(out.state["findings"], [])


class NativeFinancialIntegrationTests(unittest.TestCase):
    setup_tools = FinancialToolTests.setup_tools
    def test_native_financial_tools_and_recorded_finding(self):
        from toolkit import LLM
        from toolkit.fake_llm import FakeLLM

        _, _, out, data, tools, dispatch = self.setup_tools()
        plan = {"payee_id": "PAN:" + PAN, "classifications": [classification()]}
        calc_id = tools.calculate_tds(**plan)["calculations"][0]["calculation_id"]
        tools.calculations.clear()
        tools.plans.clear()
        script = {"steps": [
            {"tool_calls": [{"name": "get_party_summary"}]},
            {"tool_calls": [{"name": "get_party_transactions", "arguments": json.dumps({"payee_id": "PAN:" + PAN})}]},
            {"tool_calls": [{"name": "calculate_tds", "arguments": json.dumps(plan)}]},
            {"tool_calls": [{"name": "record_assessment", "arguments": json.dumps({"calculation_id": calc_id,
                "reason": "The synthetic contractor payment exceeds the single-payment threshold."})}]},
            {"content": '{"findings":[]}'}]}
        server = FakeLLM(script).start()
        self.addCleanup(server.stop)
        agent = Agent(LLM(model="local-financial-test", base_url=server.url, api_key="fake"), dispatch, out)
        reason = agent.run()
        self.assertEqual(reason, "financial_review_complete")
        out.finalize(reason, agent.metrics())
        self.assertTrue(out.state["run"]["coverage_complete"])
        self.assertEqual(out.state["findings"][0]["tds_required"], Decimal("800"))
        records = [json.loads(line) for line in out.trace_path.read_text().splitlines()]
        starts = [r for r in records if r["event"] == "tool_start" and not r["tool_call_id"].startswith("ingestion_")]
        results = [r for r in records if r["event"] == "tool_result" and not r["tool_call_id"].startswith("ingestion_")]
        self.assertEqual([r["tool_call_id"] for r in starts], [r["tool_call_id"] for r in results])
        self.assertEqual([r["name"] for r in starts], ["get_party_summary", "get_party_transactions", "calculate_tds", "record_assessment"])
        self.assertTrue(all(r["result"]["success"] for r in results))
        for entry in server.requests[1:]:
            self.assertEqual(len(entry["tool_results"]), 1)

    def test_partial_results_survive_real_fake_api_failure(self):
        from toolkit import LLM
        from toolkit.fake_llm import FakeLLM

        _, _, out, data, tools, dispatch = self.setup_tools(rows=[row("A"), row("B", party="Other")],
                                                         parties=[party(), party("Other", OTHER_PAN)])
        plan = {"payee_id": "PAN:" + PAN, "classifications": [classification(["A"])]}
        calc_id = tools.calculate_tds(**plan)["calculations"][0]["calculation_id"]
        tools.calculations.clear()
        tools.plans.clear()
        script = {"steps": [
            {"tool_calls": [{"name": "calculate_tds", "arguments": json.dumps(plan)}]},
            {"tool_calls": [{"name": "record_assessment", "arguments": json.dumps({"calculation_id": calc_id,
                "reason": "The stored synthetic contractor voucher exceeds the statutory threshold."})}]},
            {"content": '{"findings":[]}'}], "faults": [{"step": 2, "status": 500, "times": 10}]}
        server = FakeLLM(script).start()
        self.addCleanup(server.stop)
        agent = Agent(LLM(model="local-test", base_url=server.url, api_key="fake"), dispatch, out, sleep=lambda _: None)
        self.assertEqual(agent.run(), "api_failure")
        out.finalize("api_failure", agent.metrics())
        self.assertFalse(out.state["run"]["coverage_complete"])
        self.assertEqual(len(out.state["findings"]), 1)
        self.assertEqual(data.coverage["A"], "assessed")
        self.assertEqual(data.coverage["B"], "unprocessed")
        out.validator.validate(json.loads((out.directory / "findings.json").read_text()))

    def test_early_empty_final_cannot_claim_complete(self):
        from test_phase1 import ScriptLLM, payload

        _, _, out, data, tools, dispatch = self.setup_tools()
        model = ScriptLLM([payload('{"findings":[],"run":{"coverage_complete":true,"status":"complete"}}')])
        agent = Agent(model, dispatch, out)
        self.assertEqual(agent.run(), "financial_review_incomplete")
        out.finalize("financial_review_incomplete", agent.metrics())
        self.assertFalse(out.state["run"]["coverage_complete"])
        self.assertEqual(out.state["run"]["status"], "incomplete")
        self.assertIn("1 unprocessed", out.state["limitations"][0])
