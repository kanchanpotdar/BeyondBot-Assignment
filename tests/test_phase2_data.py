import csv
import unittest
from decimal import Decimal
from unittest.mock import patch

from phase2_support import fixture, row, party, PAN, OTHER_PAN, PERSON_PAN
from tds_agent.data import Engagement, money


class IngestionTests(unittest.TestCase):
    def test_all_rows_and_original_source_fields(self):
        rows = [row(f"V{i}", "0.01") for i in range(125)]
        root, kit, out, data = fixture(self, rows=rows)
        self.assertEqual(len(data.transactions), 125)
        self.assertEqual(sum(r["amount"] for r in data.transactions.values()), Decimal("1.25"))
        r = data.transactions["V0"]
        for key in ("date", "voucher_no", "narration", "ledger", "tds_section"):
            self.assertEqual(r[key], rows[0][key])
        self.assertEqual(r["raw_amount"], "0.01")
        self.assertEqual(r["source_row"], 2)
        self.assertEqual(r["source"], "books/expense_ledger.csv")
        self.assertFalse(data.coverage_report()["complete"])

    def test_empty_inputs_explicitly_incomplete(self):
        _, _, _, data = fixture(self, rows=[], parties=[])
        self.assertFalse(data.coverage_report()["complete"])
        self.assertEqual(len(data.limitations), 2)

    def test_missing_files(self):
        for path in ("books/expense_ledger.csv", "books/party_master.csv", "client.json", "rules/engagement_rules.yaml"):
            with self.subTest(path=path):
                root, kit, out, _ = fixture(self, load=False)
                (root / path).unlink()
                with self.assertRaises((FileNotFoundError, ValueError, KeyError)):
                    Engagement(root, kit, out)

    def test_missing_and_duplicate_headers_or_incomplete_record(self):
        for content in ("party,amount\nVendor,1\n", "date,date\nx,x\n",
                        "date,voucher_no,voucher_type,ledger,party,narration,amount,tds_section,tds_amount\n2025-04-01,V1,Journal,Fees,Vendor,text,1\n"):
            root, kit, out, _ = fixture(self, load=False)
            (root / "books/expense_ledger.csv").write_text(content)
            with self.assertRaises(ValueError):
                Engagement(root, kit, out)

    def test_invalid_numeric_and_null_fields(self):
        for amount in ("", "NaN", "Infinity", "not money", "-1", "0.000000001"):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                fixture(self, rows=[row(amount=amount)])
        for amount in (None, True, 0.1):
            with self.assertRaises(ValueError):
                money(amount)
        with self.assertRaises(ValueError):
            fixture(self, rows=[row(tds="bad")])

    def test_duplicate_vouchers_rejected_legitimate_repeated_payments_preserved(self):
        with self.assertRaises(ValueError):
            fixture(self, rows=[row(), row()])
        _, _, _, data = fixture(self, rows=[row("A"), row("B")])
        self.assertEqual(len(data.transactions), 2)

    def test_cross_period_and_invalid_dates(self):
        for date in ("2025-03-31", "2026-04-01", "2025-02-30"):
            with self.subTest(date=date), self.assertRaises(ValueError):
                fixture(self, rows=[row(date=date)])

    def test_pan_normalization_alias_aggregation_and_missing_pan_separation(self):
        parties = [party("Alpha", " aaafa0001a "), party("Alias", PAN), party("Alpha Ltd", OTHER_PAN),
                   party("Missing One", ""), party("Missing Two", "")]
        rows = [row(str(i), "100", party=p["party"]) for i, p in enumerate(parties)]
        _, _, _, data = fixture(self, rows=rows, parties=parties)
        self.assertEqual(len(data.payees), 4)
        self.assertEqual(data.payees["PAN:" + PAN]["vouchers"], ["0", "1"])
        self.assertEqual(len([p for p in data.payees.values() if p["pan"] is None]), 2)
        self.assertIn("PAN:" + OTHER_PAN, data.payees)

    def test_conflicting_duplicate_party_and_unmapped_party_rejected(self):
        with self.assertRaises(ValueError):
            fixture(self, parties=[party(), party(pan=OTHER_PAN)])
        with self.assertRaises(ValueError):
            fixture(self, rows=[row(party="Unknown")])
        with self.assertRaises(ValueError):
            fixture(self, parties=[party(pan="invalid")])

    def test_ledger_views_do_not_merge_distinct_pans(self):
        _, _, _, data = fixture(self, rows=[row("A", party="One"), row("B", party="Two")],
                                 parties=[party("One", PAN), party("Two", OTHER_PAN)])
        self.assertEqual(len(data.groups), 2)
        with self.assertRaises(ValueError):
            data.select(["A", "B"])

    def test_pagination_boundaries_and_unknown_entities(self):
        _, _, _, data = fixture(self, rows=[row(str(i)) for i in range(101)])
        first = data.get_party_transactions("PAN:" + PAN, limit=100)
        self.assertEqual(first["total"], 101)
        self.assertEqual(first["next_offset"], 100)
        second = data.get_party_transactions("PAN:" + PAN, offset=100, limit=100)
        self.assertEqual(len(second["items"]), 1)
        self.assertIsNone(second["next_offset"])
        self.assertEqual(data.get_party_transactions("PAN:" + PAN, offset=101)["items"], [])
        with self.assertRaises(ValueError):
            data.get_party_summary("missing")

    def test_complete_document_pagination_and_caching(self):
        text = "Evidence\n" + "X" * 12001
        _, kit, _, data = fixture(self, documents={"long": text})
        with patch.dict(kit.functions, {"read_document": kit.read_document}):
            with patch.object(data, "call", wraps=data.call) as calls:
                self.assertEqual(data.read_document("long"), text)
                self.assertEqual(data.read_document("long"), text)
                self.assertEqual(calls.call_count, 3)

    def test_incomplete_and_nonprogressing_document_pages_rejected(self):
        from toolkit.envelope import ok
        for data_result in ({"offset": 0, "total_chars": 8, "text": "abc", "next_offset": None},
                            {"offset": 0, "total_chars": 8, "text": "", "next_offset": 0}):
            _, kit, _, data = fixture(self, documents={"evidence": "12345678"})
            with patch.dict(kit.functions, {"read_document": lambda **_: ok(data_result)}):
                with self.assertRaises(ValueError):
                    data.read_document("evidence")

    def test_rule_discovery_is_by_year_not_filename(self):
        root, kit, out, _ = fixture(self, load=False)
        old = root / "rules/engagement_rules.yaml"
        old.rename(root / "rules/renamed.yaml")
        self.assertEqual(Engagement(root, kit, out).rules_id, "rules/renamed.yaml")
        (root / "rules/duplicate.yaml").write_text((root / "rules/renamed.yaml").read_text())
        with self.assertRaises(ValueError):
            Engagement(root, kit, out)

    def test_large_summary_preserves_decimal_precision(self):
        _, _, _, data = fixture(self, rows=[row("A", "100000000000000000000.00000001"), row("B", "0.00000001")])
        summary = data.get_party_summary()["items"][0]
        self.assertEqual(summary["amount"], Decimal("100000000000000000000.00000002"))
