"""Expected rupee values below are independently specified, never engine-derived."""
import unittest
from decimal import Decimal

from phase2_support import fixture, row, party, cert, transporter, turnover, PAN, PERSON_PAN
from tds_agent.calculations import Calculator
from tds_agent.output import exact_json


class FinancialTests(unittest.TestCase):
    def calculate(self, rows, section, category, **kwargs):
        fixture_keys = {k: kwargs.pop(k) for k in list(kwargs) if k in ("parties", "documents", "rule_change")}
        _, _, _, data = fixture(self, rows=rows, **fixture_keys)
        return Calculator(data).calculate(list(data.transactions), section, category, [rows[0]["voucher_no"]], **kwargs)

    def test_194c_single_threshold_below_at_above(self):
        for amount, base, required in (("29999.99", "0", "0"), ("30000", "0", "0"),
                                       ("30000.01", "30000.01", "600.00")):
            with self.subTest(amount=amount):
                result = self.calculate([row(amount=amount)], "194C", "contractor")
                self.assertEqual(result["amount_liable"], Decimal(base))
                self.assertEqual(result["tds_required"], Decimal(required))

    def test_194c_annual_threshold_below_at_above_and_whole_amount(self):
        for last, base, required in (("24999.99", "0", "0"), ("25000", "0", "0"),
                                     ("25000.01", "100000.01", "2000.00")):
            rows = [row(str(i), "25000") for i in range(3)] + [row("last", last)]
            with self.subTest(last=last):
                r = self.calculate(rows, "194C", "contractor")
                self.assertEqual(r["amount_liable"], Decimal(base))
                self.assertEqual(r["tds_required"], Decimal(required))
        r = self.calculate([row("A", "40000"), row("B", "10000")], "194C", "contractor")
        self.assertEqual(r["amount_liable"], Decimal("40000"))
        self.assertEqual(r["tds_required"], Decimal("800"))

    def test_rates_individual_huf_and_missing_pan(self):
        for pan, expected in ((PERSON_PAN, "400"), ("AAAHA0001A", "400"), (PAN, "800"), ("", "8000")):
            with self.subTest(pan=pan):
                kind = "Individual" if pan == PERSON_PAN else ("HUF" if pan == "AAAHA0001A" else "Firm")
                result = self.calculate([row()], "194C", "contractor", parties=[party(pan=pan, entity_type=kind)])
                self.assertEqual(result["tds_required"], Decimal(expected))

    def test_conflicting_pan_entity_type_is_unresolved(self):
        r = self.calculate([row()], "194C", "contractor", parties=[party(pan=PERSON_PAN, entity_type="Company")])
        self.assertEqual(r["status"], "needs_review")
        self.assertIn("PAN/entity-type", r["needs"])

    def test_operational_errors_cannot_be_financial_needs_review(self):
        with self.assertRaises(ValueError):
            self.calculate([row()], "unknown", "unknown", needs="The API failed and the payee is unprocessed")

    def test_annual_thresholds_194h_194a_and_194j(self):
        cases = [("194H", "commission", "19999.99", "20000", "20000.01", "400.00"),
                 ("194A", "interest", "9999.99", "10000", "10000.01", "1000.00"),
                 ("194J", "professional", "49999.99", "50000", "50000.01", "5000.00"),
                 ("194J", "technical", "49999.99", "50000", "50000.01", "1000.00"),
                 ("194J", "royalty", "49999.99", "50000", "50000.01", "5000.00"),
                 ("194J", "non_compete", "49999.99", "50000", "50000.01", "5000.00")]
        for section, category, below, at, above, due in cases:
            for amount, expected in ((below, "0"), (at, "0"), (above, due)):
                with self.subTest(section=section, category=category, amount=amount):
                    result = self.calculate([row(amount=amount)], section, category)
                    self.assertEqual(result["tds_required"], Decimal(expected))

    def test_director_zero_threshold(self):
        for amount, expected in (("0", "0"), ("0.01", "0"), ("1", "0.10"), ("15000", "1500")):
            r = self.calculate([row(amount=amount)], "194J", "director")
            self.assertEqual(r["tds_required"], Decimal(expected))
            self.assertEqual(r["amount_liable"], Decimal(amount))

    def test_monthly_rent_threshold_and_month_separation(self):
        for amount, expected in (("49999.99", "0"), ("50000", "0"), ("50000.01", "5000.00")):
            result = self.calculate([row(amount=amount)], "194I", "land_building_furniture")
            self.assertEqual(result["tds_required"], Decimal(expected))
        rows = [row("A", "30000"), row("B", "25000", date="2025-04-30"),
                row("C", "50000", date="2025-05-01")]
        r = self.calculate(rows, "194I", "land_building_furniture")
        self.assertEqual(r["amount_liable"], Decimal("55000"))
        self.assertEqual(r["tds_required"], Decimal("5500"))
        r = self.calculate([row(amount="60000")], "194I", "plant_machinery_equipment")
        self.assertEqual(r["tds_required"], Decimal("1200"))

    def test_no_pan_floor_for_other_sections(self):
        r = self.calculate([row(amount="60000")], "194J", "technical", parties=[party(pan="")])
        self.assertEqual(r["tds_required"], Decimal("12000"))

    def test_cert_validity_and_cap_split(self):
        rows = [row("before", "40000", date="2025-09-30"), row("start", "40000", date="2025-10-01"),
                row("cap", "40000", date="2025-11-01")]
        r = self.calculate(rows, "194J", "professional", documents={"certificate": cert()}, certificate_ids=["certificate"])
        self.assertEqual(r["tds_required"], Decimal("6600.00"))
        self.assertEqual([p["base"] for p in r["breakdown"][2]["rates"]], [Decimal("20000"), Decimal("20000")])
        rows = [row("start", "40000", date="2025-10-01"), row("end", "40000", date="2025-10-31"),
                row("after", "40000", date="2025-11-01")]
        r = self.calculate(rows, "194J", "professional", documents={"certificate": cert(cap="100000", end="31-10-2025")}, certificate_ids=["certificate"])
        self.assertEqual(r["tds_required"], Decimal("4800"))

    def test_certificate_payer_pan_category_and_bad_evidence(self):
        for certificate in (cert(tan="PNEB99999A"), cert(pan=PERSON_PAN), cert(section="194J — technical services")):
            r = self.calculate([row(amount="100000", date="2025-10-01")], "194J", "professional",
                               documents={"certificate": certificate}, certificate_ids=["certificate"])
            self.assertEqual(r["tds_required"], Decimal("10000"))
        r = self.calculate([row(amount="100000")], "194J", "professional",
                           documents={"certificate": "CERTIFICATE\nRate: bad"}, certificate_ids=["certificate"])
        self.assertEqual(r["status"], "needs_review")
        self.assertEqual(r["shortfall"], 0)
        self.assertTrue(r["needs"])
        r = self.calculate([row(amount="100000")], "194J", "professional",
                           documents={"certificate": cert(cap="6,,00,000")}, certificate_ids=["certificate"])
        self.assertEqual(r["status"], "needs_review")
        self.assertIn("malformed digit grouping", r["needs"])
        with self.assertRaises(ValueError):
            self.calculate([row()], "194C", "contractor", certificate_ids=["missing"])

    def test_transporter_exemption_conditions(self):
        for document, expected in ((transporter(), "0"), (transporter(maximum="eleven", count=11), "800"),
                                   (transporter(fy="2024-25"), "800"), (transporter(company="Other Buyer"), "800")):
            r = self.calculate([row()], "194C", "contractor", documents={"declaration": document}, transporter_id="declaration")
            self.assertEqual(r["tds_required"], Decimal(expected))
        r = self.calculate([row()], "194C", "contractor", parties=[party(pan="")],
                           documents={"declaration": transporter()}, transporter_id="declaration")
        self.assertEqual(r["tds_required"], Decimal("8000"))
        r = self.calculate([row()], "194C", "contractor", documents={"declaration": transporter(count=11)}, transporter_id="declaration")
        self.assertEqual(r["status"], "needs_review")

    def test_194q_turnover_below_at_above_condition(self):
        for buyer, expected in (("99999999.99", "0"), ("100000000", "0"), ("100000000.01", "100")):
            r = self.calculate([row(amount="5100000")], "194Q", "goods", documents={"turnover": turnover(buyer)}, turnover_id="turnover")
            self.assertEqual(r["tds_required"], Decimal(expected))
        for amount, base in (("4999999.99", "0"), ("5000000", "0"), ("5000000.01", "0.01")):
            r = self.calculate([row(amount=amount)], "194Q", "goods", documents={"turnover": turnover()}, turnover_id="turnover")
            self.assertEqual(r["amount_liable"], Decimal(base))

    def test_194q_excess_is_chronological_and_missing_pan_rate(self):
        rows = [row("A", "4900000"), row("B", "200000", date="2025-05-01"), row("C", "100000", date="2025-06-01")]
        r = self.calculate(rows, "194Q", "goods", documents={"turnover": turnover()}, turnover_id="turnover")
        self.assertEqual(r["amount_liable"], Decimal("200000"))
        self.assertEqual(r["tds_required"], Decimal("200"))
        self.assertEqual([b["amount_liable"] for b in r["breakdown"]], [Decimal(0), Decimal("100000"), Decimal("100000")])
        r = self.calculate(rows, "194Q", "goods", parties=[party(pan="")], documents={"turnover": turnover()}, turnover_id="turnover")
        self.assertEqual(r["tds_required"], Decimal("10000"))

    def test_194q_missing_wrong_year_or_wrong_buyer_is_unresolved(self):
        r = self.calculate([row(amount="5100000")], "194Q", "goods")
        self.assertEqual(r["status"], "needs_review")
        self.assertEqual(r["shortfall"], 0)
        for document in (turnover(fy="2023-24"), turnover(company="Other")):
            r = self.calculate([row(amount="5100000")], "194Q", "goods", documents={"turnover": document}, turnover_id="turnover")
            self.assertEqual(r["status"], "needs_review")

    def test_certificate_cap_intersects_only_trailing_194q_excess(self):
        rows = [row("A", "4900000"), row("B", "200000", date="2025-05-01")]
        documents = {"turnover": turnover(), "certificate": cert(rate="0.05", cap="5050000", start="01-04-2025", section="194Q — goods")}
        r = self.calculate(rows, "194Q", "goods", documents=documents, turnover_id="turnover", certificate_ids=["certificate"])
        # First 4.9m consumes the cap without tax. Next 100k reaches threshold;
        # the remaining 100k has 50k at 0.05% (25), then 50k at 0.1% (50).
        self.assertEqual(r["amount_liable"], Decimal("100000"))
        self.assertEqual(r["tds_required"], Decimal("75"))

    def test_disjoint_certificates_supported_overlapping_priority_unresolved(self):
        rows = [row("A", "40000", date="2025-04-01"), row("B", "40000", date="2025-08-01")]
        documents = {"first": cert(rate="1", cap="20000", start="01-04-2025", end="31-07-2025"),
                     "second": cert(rate="2", cap="40000", start="01-08-2025", end="31-12-2025")}
        r = self.calculate(rows, "194J", "professional", documents=documents, certificate_ids=["first", "second"])
        self.assertEqual(r["tds_required"], Decimal("3000"))
        documents["second"] = cert(rate="2", start="01-07-2025")
        r = self.calculate(rows, "194J", "professional", documents=documents, certificate_ids=["first", "second"])
        self.assertEqual(r["status"], "needs_review")
        self.assertIn("overlapping", r["needs"])

    def test_decimal_rounding_and_consistency(self):
        r = self.calculate([row(amount="20000.25", tds="399.995")], "194H", "commission")
        self.assertEqual(r["tds_required"], Decimal("400.01"))
        self.assertEqual(r["tds_deducted"], Decimal("400.00"))
        self.assertEqual(r["shortfall"], Decimal("0.01"))
        r = self.calculate([row("A", "0.05"), row("B", "0.05")], "194J", "director")
        self.assertEqual(r["tds_required"], Decimal("0.01"))  # Round total, not each 0.005 segment.
        self.assertIn('"tds_required":0.01', exact_json(r))
        r = self.calculate([row(amount="100000000000000000000")], "194J", "director")
        self.assertEqual(r["tds_required"], Decimal("10000000000000000000"))

    def test_deduction_statuses_and_no_negative_shortfall(self):
        for tds, status, shortfall in (("0", "not_deducted", "800"), ("100", "short_deducted", "700"),
                                       ("800", "compliant", "0"), ("1000", "compliant", "0")):
            r = self.calculate([row(tds=tds)], "194C", "contractor")
            self.assertEqual((r["status"], r["shortfall"]), (status, Decimal(shortfall)))

    def test_yaml_rates_thresholds_are_authoritative(self):
        def change(rules):
            rules["sections"]["194C"]["rate_others"] = "3"
            rules["sections"]["194C"]["threshold_single_payment"] = "40000"
        r = self.calculate([row(amount="40000.01")], "194C", "contractor", rule_change=change)
        self.assertEqual(r["tds_required"], Decimal("1200.00"))
        r = self.calculate([row(amount="40000")], "194C", "contractor", rule_change=change)
        self.assertEqual(r["tds_required"], 0)

    def test_rule_missing_or_unsupported_is_not_compliant(self):
        def change(rules):
            del rules["sections"]["194C"]["rate_others"]
        r = self.calculate([row()], "194C", "contractor", rule_change=change)
        self.assertEqual(r["status"], "needs_review")
        self.assertIn("rate_others", r["needs"])
        def additional_rule(rules):
            rules["sections"]["194C"]["new_eligibility_condition"] = "Unsupported condition"
        r = self.calculate([row()], "194C", "contractor", rule_change=additional_rule)
        self.assertEqual(r["status"], "needs_review")
        self.assertIn("new_eligibility_condition", r["needs"])

    def test_unknown_financial_fact_and_excluded_cases(self):
        r = self.calculate([row()], "unknown", "unknown", needs="Confirm whether this payment buys goods or services")
        self.assertEqual(r["status"], "needs_review")
        self.assertNotIn("tds_required", r)
        r = self.calculate([row()], "excluded", "excluded")
        self.assertEqual((r["status"], r["shortfall"]), ("excluded", Decimal(0)))
