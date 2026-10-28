"""The offline audit independently rejects invented amounts and lost coverage."""
import json
import copy
import tempfile
import unittest
from pathlib import Path

from scripts.audit_financial_run import audit


class FinancialAuditTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "books").mkdir()
        (root / "books/expense_ledger.csv").write_text("voucher_no,date,amount,tds_amount\nV1,2025-04-01,100,0\n")
        calculation = {"party": "Synthetic", "section": "194C", "status": "not_deducted", "vouchers": ["V1"],
                       "amount_total": 100, "tds_deducted": 0, "tds_required": 1, "shortfall": 1, "amount_liable": 100,
                       "breakdown": [{"voucher_no": "V1", "date": "2025-04-01", "amount": 100,
                                      "tds_deducted": 0, "amount_liable": 100, "tds_required_exact": 1,
                                      "rates": [{"base": 100, "rate": 1}]}]}
        finding = {k: v for k, v in calculation.items() if k not in ("breakdown", "amount_total")}
        output = {"run": {"run_id": "test"}, "findings": [finding]}
        records = [{"event": "financial_calculation", "calculation": calculation},
                   {"event": "assessment_recorded", "assessment": calculation},
                   {"event": "coverage_final", "unprocessed_vouchers": [], "unresolved_vouchers": []}]
        for r in records:
            r["run_id"] = "test"
        return root, output, records

    def save(self, root, output, records):
        (root / "findings.json").write_text(json.dumps(output))
        (root / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    def test_independent_one_rupee_expectation(self):
        root, output, records = self.fixture()
        self.save(root, output, records)
        self.assertTrue(audit(root, root)["arithmetic_and_partition_valid"])

    def test_invented_finding_amount_rejected(self):
        root, output, records = self.fixture()
        output["findings"][0]["shortfall"] = 99
        self.save(root, output, records)
        with self.assertRaises(AssertionError):
            audit(root, root)

    def test_untracked_source_voucher_rejected(self):
        root, output, records = self.fixture()
        records.pop(1)
        output["findings"] = []
        self.save(root, output, records)
        with self.assertRaises(AssertionError):
            audit(root, root)

    def test_existing_excluded_contract_requires_zero_liability_and_source_totals(self):
        root, output, records = self.fixture()
        calculation = records[0]['calculation']
        calculation.update(section='excluded', category='excluded', status='excluded', amount_liable=0,
                           tds_required=0, shortfall=0, breakdown=[])
        output['findings'] = []
        self.save(root, output, records)
        self.assertTrue(audit(root, root)['arithmetic_and_partition_valid'])
        calculation['tds_required'] = 1
        self.save(root, output, records)
        with self.assertRaises(AssertionError):
            audit(root, root)

    def test_identical_calculation_replay_counts_once_but_changed_values_fail(self):
        root, output, records = self.fixture()
        records[0]['calculation']['calculation_id'] = 'stored-one'
        replay = copy.deepcopy(records[1])
        records.insert(2, replay)
        self.save(root, output, records)
        result = audit(root, root)
        self.assertEqual(result['recorded_vouchers'], 1)
        self.assertEqual(result['replayed_assessment_events'], 1)
        replay['assessment']['shortfall'] = 99
        self.save(root, output, records)
        with self.assertRaises(AssertionError):
            audit(root, root)
