"""Section-specific arithmetic governed only by the engagement YAML.

Rounding policy (the assessment specifies none): sum exact segment liabilities,
then round each payee/section/category assessment HALF_UP to paise. Shortfall is
computed from rounded required and deducted totals. No interest or penalties.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP, localcontext

from .data import money

ZERO = Decimal(0)
CENT = Decimal("0.01")
CATEGORIES = {"194C": {"contractor"}, "194H": {"commission"}, "194A": {"interest"},
              "194Q": {"goods"}, "194I": {"plant_machinery_equipment", "land_building_furniture"},
              "194J": {"professional", "technical", "royalty", "non_compete", "director"},
              "excluded": {"excluded"}, "unknown": {"unknown"}}
RULE_KEYS = {
    "194C": {"nature", "threshold_single_payment", "threshold_yearly", "rate_individual_huf", "rate_others"},
    "194H": {"nature", "threshold_yearly", "rate"},
    "194A": {"nature", "threshold_yearly_payer_not_bank", "rate"},
    "194I": {"nature", "threshold_monthly", "rate_plant_machinery_equipment", "rate_land_building_furniture"},
    "194J": {"nature", "categories", "threshold_yearly_per_category", "threshold_director", "rate_technical_services", "rate_others"},
    "194Q": {"nature", "applies_only_if", "buyer_turnover_condition", "threshold_yearly_per_seller", "rate_on_excess"}}


class Unresolved(ValueError):
    """A specific financial fact or supported rule interpretation is missing."""


def rounded(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def source_date(value):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d %B %Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    raise Unresolved("Confirm the certificate validity dates in a supported unambiguous date format")


def field(text, label):
    matches = re.findall(r"^" + label + r"\s*:\s*(.+)$", text, flags=re.M | re.I)
    if len(matches) != 1:
        raise Unresolved(f"Confirm the single {label} field in the source document")
    return matches[0].strip()


def amount_text(text):
    match = re.fullmatch(r"(?:Rs\.?\s*)?([0-9][0-9,]*(?:\.[0-9]+)?)", text.strip(), re.I)
    if not match:
        raise Unresolved("Confirm the source amount as an unambiguous rupee value")
    integer = match[1].split(".")[0]
    if "," in integer and not (re.fullmatch(r"[1-9]\d{0,2}(?:,\d{3})+", integer) or
                                re.fullmatch(r"[1-9]\d?(?:,\d{2})*,\d{3}", integer)):
        raise Unresolved("Resolve malformed digit grouping in the source amount")
    return money(match[1].replace(",", ""))


def parse_certificate(text, doc_id, client):
    """Narrow label-based extraction; model-supplied rates/caps are never inputs."""
    if not re.search(r"CERTIFICATE UNDER SECTION 197\b", text, re.I):
        raise Unresolved("Confirm the source is a certificate under section 197")
    deductee, deductor = field(text, "Deductee"), field(text, "Deductor")
    pan = re.search(r"\bPAN\s+([A-Z]{5}[0-9]{4}[A-Z])\b", deductee)
    tan = re.search(r"\bTAN\s+([A-Z]{4}[0-9]{5}[A-Z])\b", deductor)
    if not pan or not tan:
        raise Unresolved("Confirm certificate deductee PAN and deductor TAN")
    if not client.get("tan"):
        raise Unresolved("Confirm this payer's TAN before applying a lower-deduction certificate")
    if tan[1] == client["tan"] and not deductor.startswith(client["company"] + " (TAN "):
        raise Unresolved("Resolve conflicting payer name and TAN on the certificate")
    scope = field(text, "Section")
    match = re.match(r"(194[A-Z])\b", scope)
    rate = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:per cent|%)", field(text, "Rate of deduction"), re.I)
    if not match or not rate:
        raise Unresolved("Confirm certificate section and deduction rate")
    category = None
    if match[1] == "194J":
        categories = [c for c in CATEGORIES["194J"] if c.replace("_", "-") in scope.lower()]
        if len(categories) != 1:
            raise Unresolved("Confirm which 194J category the certificate covers")
        category = categories[0]
    start, end = source_date(field(text, "Valid from")), source_date(field(text, "Valid to"))
    if end < start:
        raise Unresolved("Resolve reversed certificate validity dates")
    parsed_rate = money(rate[1])
    if parsed_rate > 100:
        raise Unresolved("Resolve certificate rate outside percentage-point bounds")
    return {"doc_id": doc_id, "pan": pan[1], "payer_tan": tan[1],
            "payer_matches": tan[1] == client.get("tan"), "section": match[1], "category": category,
            "rate": parsed_rate, "from": start, "to": end,
            "cap": amount_text(field(text, "Amount up to which this certificate is valid"))}


def integer_word(value):
    words = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    return int(value) if value.isdigit() else words.index(value.lower())


def transporter_exempt(text, doc_id, data, pan):
    rule = data.rules.get("transporter_194C6", "")
    allowed = re.search(r"([a-z]+|\d+) or fewer goods carriages at any time", rule, re.I)
    declared = re.search(r"not own more than ([a-z]+|\d+) goods carriages at any time during the previous year\s+([0-9-]+)", text, re.I)
    if not allowed or not declared:
        raise Unresolved("Confirm year-wide transporter ownership conditions and declaration")
    try:
        limit, maximum = integer_word(allowed[1]), integer_word(declared[1])
    except ValueError as exc:
        raise Unresolved("Unsupported vehicle-count expression in the rule or declaration") from exc
    total = re.search(r"Total\s*:\s*(\d+) goods carriages", text, re.I)
    payer = field(text, "To")
    if not text.upper().count(f"PAN {pan}") or not pan:
        return False
    company = data.client["company"]
    if not (payer == company or payer.startswith(company + ",")) or declared[2] != str(data.rules["fy"]):
        return False
    if not re.search(r"business of plying, hiring and leasing goods carriages", text, re.I):
        raise Unresolved("Confirm that the contractor operates goods carriages")
    if not total or int(total[1]) > maximum:
        raise Unresolved("Resolve inconsistent transporter vehicle count and ownership declaration")
    return maximum <= limit


def prior_turnover(text, client, current_fy):
    """Only an explicitly labeled, source-bound prior-year business figure."""
    previous_start = int(current_fy[:4]) - 1
    previous_fy = f"{previous_start}-{str(previous_start + 1)[-2:]}"
    if field(text, "Company") != client["company"] or field(text, "Financial year") != previous_fy:
        raise Unresolved("Confirm turnover belongs to this buyer and the immediately preceding financial year")
    return amount_text(field(text, "Business turnover"))


class Calculator:
    def __init__(self, data):
        self.data, self.rules = data, data.rules

    def number(self, section, key, refs):
        refs.add(f"{self.data.rules_id}#sections.{section}.{key}")
        try:
            return money(self.rules["sections"][section][key])
        except (KeyError, ValueError) as exc:
            raise Unresolved(f"Unsupported or missing rule: sections.{section}.{key}") from exc

    def calculate(self, vouchers, section, category, evidence, needs=None,
                  certificate_ids=None, transporter_id=None, turnover_id=None, rent_vouchers=None):
        with localcontext() as context:
            context.prec = 60
            return self._calculate(vouchers, section, category, evidence, needs,
                                   certificate_ids or [], transporter_id, turnover_id, rent_vouchers)

    def _calculate(self, vouchers, section, category, evidence, needs, certificate_ids, transporter_id, turnover_id, rent_vouchers):
        if section not in CATEGORIES or category not in CATEGORIES[section]:
            raise ValueError("Unsupported section/category classification")
        rows = self.data.select(vouchers)
        identity = rows[0]["payee_id"]
        payee = self.data.payees[identity]
        pan = payee["pan"]
        total = sum((r["amount"] for r in rows), ZERO)
        deducted = rounded(sum((r["tds_amount"] for r in rows), ZERO))
        references = {self.data.rules_id + "#how_to_apply"}
        sources = list(dict.fromkeys(evidence + certificate_ids + ([transporter_id] if transporter_id else []) + ([turnover_id] if turnover_id else [])))
        if not sources:
            raise ValueError("Classification requires source evidence")
        for source in sources:
            if source in self.data.transactions:
                if source not in vouchers:
                    raise ValueError("Classification evidence voucher belongs to another calculation")
            elif source in self.data.review_evidence:
                pass
            else:
                self.data.read_document(source)
        base_result = {"payee_id": identity, "party": rows[0]["party"], "parties": sorted({r["party"] for r in rows}),
                       "pan": pan, "section": section, "category": category,
                       "vouchers": [r["voucher_no"] for r in rows], "evidence": sources,
                       "aggregation_period": {"from": str(self.data.start), "to": str(self.data.end)},
                       "amount_total": total, "tds_deducted": deducted,
                       "rounding": "Exact segment sum, then ROUND_HALF_UP to 0.01 per payee/section/category",
                       "assumptions": [], "breakdown": []}
        try:
            if payee.get("identity_issues"):
                raise Unresolved("Resolve PAN/entity-type conflicts in the party master before financial assessment")
            if section == "unknown":
                if not needs or not needs.strip():
                    raise ValueError("Unresolved assessment requires a specific financial needs statement")
                if re.search(r"\b(api|timeout|model|unprocessed)\b|tool error|rate limit", needs, re.I):
                    raise ValueError("Operational failures cannot be recorded as unresolved financial facts")
                raise Unresolved(needs)
            if section == "excluded":
                return {**base_result, "amount_liable": ZERO, "tds_required": ZERO, "shortfall": ZERO,
                        "status": "excluded", "rule_references": ["PRIMER.md#out-of-scope-expenses"]}
            s = self.rules.get("sections", {}).get(section)
            if not isinstance(s, dict):
                raise Unresolved(f"Missing supported rules for {section}")
            extra = set(s) - RULE_KEYS[section]
            if extra:
                raise Unresolved(f"Confirm unsupported {section} rule clauses: {', '.join(sorted(extra))}")
            bases = {r["voucher_no"]: ZERO for r in rows}
            if section == "194C":
                annual = self.number(section, "threshold_yearly", references)
                single = self.number(section, "threshold_single_payment", references)
                for r in rows:
                    if total > annual or r["amount"] > single:
                        bases[r["voucher_no"]] = r["amount"]
                rate_key = "rate_individual_huf" if pan and pan[3] in "PH" else "rate_others"
                # Missing PAN takes max(ordinary rate, no-PAN rate), so the
                # unknown entity type cannot silently lower its deduction.
            elif section in ("194H", "194A", "194J"):
                key = {"194H": "threshold_yearly", "194A": "threshold_yearly_payer_not_bank",
                       "194J": "threshold_director" if category == "director" else "threshold_yearly_per_category"}[section]
                if section == "194A" and "not a bank" not in self.data.client["constitution"].lower():
                    raise Unresolved("Confirm the payer is not a bank; only non-bank 194A rules are supplied")
                if section == "194J" and category not in s.get("categories", []):
                    raise Unresolved("Category not supported by the engagement 194J rules")
                if total > self.number(section, key, references):
                    bases = {r["voucher_no"]: r["amount"] for r in rows}
                rate_key = "rate" if section != "194J" else ("rate_technical_services" if category == "technical" else "rate_others")
            elif section == "194I":
                months = defaultdict(lambda: ZERO)
                rent_rows = self.data.select(rent_vouchers) if rent_vouchers else rows
                if rent_rows[0]["payee_id"] != identity:
                    raise ValueError("Rent aggregation cannot cross payee identities")
                for r in rent_rows:
                    months[r["date"][:7]] += r["amount"]
                for r in rows:
                    if months[r["date"][:7]] > self.number(section, "threshold_monthly", references):
                        bases[r["voucher_no"]] = r["amount"]
                rate_key = "rate_" + category
                base_result["assumptions"].append("Rent threshold is tested across all rent categories per payee/month; rate follows each category")
            else:  # 194Q: threshold applies to seller goods only; excess accrues chronologically.
                threshold = self.number(section, "threshold_yearly_per_seller", references)
                eligible = False
                if total > threshold:
                    if not turnover_id:
                        raise Unresolved("Confirm the buyer's immediately preceding-year business turnover for 194Q eligibility")
                    condition = self.number(section, "buyer_turnover_condition", references)
                    bound_fact = self.data.eligibility_facts.get(turnover_id)
                    if bound_fact:
                        if bound_fact["condition"] != condition:
                            raise Unresolved("Revalidate turnover evidence against the changed YAML eligibility condition")
                        eligible = bound_fact["eligible"]
                        base_result["eligibility"] = {"source": turnover_id, "eligible": eligible, "verified_bound": bound_fact}
                    else:
                        text = self.data.review_evidence.get(turnover_id) or self.data.read_document(turnover_id)
                        turnover = prior_turnover(text, self.data.client, str(self.rules["fy"]))
                        eligible = turnover > condition
                        base_result["eligibility"] = {"business_turnover": turnover, "source": turnover_id, "eligible": eligible}
                consumed = ZERO
                if eligible:
                    for r in rows:
                        before = max(ZERO, consumed - threshold)
                        consumed += r["amount"]
                        bases[r["voucher_no"]] = max(ZERO, consumed - threshold) - before
                rate_key = "rate_on_excess"
            rate = self.number(section, rate_key, references)
            references.add(self.data.rules_id + "#non_filer_206AB")
            if "Omitted" not in str(self.rules.get("non_filer_206AB", "")):
                raise Unresolved("Unsupported non-filer rule structure; confirm applicable treatment")
            if rate > 100:
                raise Unresolved("Rule rate exceeds supported percentage-point bounds")
            if not pan:
                key = "rate_194Q" if section == "194Q" else "rate"
                try:
                    floor = money(self.rules["no_pan_206AA"][key])
                except (KeyError, ValueError) as exc:
                    raise Unresolved("Confirm the missing-PAN rate in the engagement rules") from exc
                references.add(f"{self.data.rules_id}#no_pan_206AA.{key}")
                rate = max(rate, floor)
                if rate > 100:
                    raise Unresolved("Missing-PAN rate exceeds supported percentage-point bounds")
            if transporter_id:
                if section != "194C":
                    raise ValueError("Transporter declarations can only apply to 194C")
                if pan and transporter_exempt(self.data.read_document(transporter_id), transporter_id, self.data, pan):
                    bases = {v: ZERO for v in bases}
                    references.add(self.data.rules_id + "#transporter_194C6")
                    base_result["assumptions"].append("Valid PAN-backed year-wide transporter declaration removes 194C liability")
            certificates = []
            for doc_id in certificate_ids:
                cert = parse_certificate(self.data.read_document(doc_id), doc_id, self.data.client)
                if cert["pan"] != pan or not cert["payer_matches"] or cert["section"] != section or (cert["category"] and cert["category"] != category):
                    base_result["assumptions"].append(f"Certificate {doc_id} does not match PAN/payer/section/category; normal rate applies")
                    continue
                if cert["rate"] > rate:
                    raise Unresolved("Resolve a lower-deduction certificate rate exceeding the ordinary applicable rate")
                certificates.append(cert)
                references.add(self.data.rules_id + "#lower_deduction_certificate_197")
            for i, first in enumerate(certificates):
                for second in certificates[i + 1:]:
                    if max(first["from"], second["from"]) <= min(first["to"], second["to"]):
                        raise Unresolved("Resolve overlapping certificates and their priority/cap interaction")
            # Allocate certificates over credits chronologically, not just liable
            # amounts. Mixed-category rent shares one section-wide cap; 194Q's
            # excess is the trailing portion of a threshold-crossing credit.
            certificate_rows = self.data.select(rent_vouchers) if section == "194I" and rent_vouchers else rows
            covered = {}
            for cert in certificates:
                cap_left = cert["cap"]
                for credit in certificate_rows:
                    if cert["from"] <= date.fromisoformat(credit["date"]) <= cert["to"]:
                        credit_covered = min(cap_left, credit["amount"])
                        cap_left -= credit_covered
                        covered[credit["voucher_no"]] = (credit_covered, cert["rate"])
            required, liable = ZERO, ZERO
            for r in rows:
                base = bases[r["voucher_no"]]
                liable += base
                portions = []
                reduced_base = ZERO
                # A certificate cap is consumed by all within-validity credits,
                # including credits below a threshold; no later reuse of its cap.
                if r["voucher_no"] in covered:
                    credit_covered, certificate_rate = covered[r["voucher_no"]]
                    reduced_base = max(ZERO, credit_covered - (r["amount"] - base))
                    if reduced_base:
                        portions.append({"base": reduced_base, "rate": certificate_rate})
                if base > reduced_base:
                    portions.append({"base": base - reduced_base, "rate": rate})
                due = sum((p["base"] * p["rate"] / Decimal(100) for p in portions), ZERO)
                required += due
                base_result["breakdown"].append({"voucher_no": r["voucher_no"], "date": r["date"],
                                                "source_row": r["source_row"], "amount": r["amount"],
                                                "amount_liable": base, "rates": portions, "tds_required_exact": due,
                                                "tds_deducted": r["tds_amount"], "period": r["date"][:7] if section == "194I" else str(self.rules["fy"])})
            required = rounded(required)
            shortfall = max(ZERO, required - deducted)
            status = "compliant" if shortfall == 0 else ("not_deducted" if deducted == 0 else "short_deducted")
            return {**base_result, "amount_liable": liable, "tds_required": required, "shortfall": shortfall,
                    "status": status, "rule_references": sorted(references)}
        except Unresolved as exc:
            return {**base_result, "status": "needs_review", "shortfall": ZERO, "needs": str(exc),
                    "rule_references": sorted(references)}
