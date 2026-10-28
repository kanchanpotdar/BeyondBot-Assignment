"""Full engagement loading, exact values, document paging and coverage identities."""
from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

import yaml

from .output import strict_json


def money(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("Monetary input must be a decimal string, not a float/null")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid monetary value") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Non-finite/negative amounts and reversal entries are unsupported")
    if result.as_tuple().exponent < -8 or result > Decimal('1e24'):
        raise ValueError("Monetary precision/magnitude exceeds supported bounds")
    return result


def pan_id(value):
    pan = re.sub(r"\s+", "", value or "").upper()
    if pan and not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan):
        raise ValueError("Invalid PAN format; entity mapping is unresolved")
    return pan or None


def money_sum(values):
    with localcontext() as context:
        context.prec = 60
        return sum(values, Decimal(0))


def page(items, offset=0, limit=20):
    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 100:
        raise ValueError("Pagination requires offset >= 0 and limit between 1 and 100")
    return {"items": items[offset:offset + limit], "total": len(items),
            "next_offset": offset + limit if offset + limit < len(items) else None}


class Engagement:
    def __init__(self, data_dir, kit, output=None):
        self.root, self.kit, self.output = Path(data_dir).resolve(), kit, output
        self._calls = 0
        self.documents = {}
        self.review_evidence = {}
        self.validated_documents = {}
        self.eligibility_facts = {}
        self.coverage = {}
        self.assignments = {}
        self.assessments = {}
        self.limitations = []
        discovery = self.call("list_tables")
        self.client = strict_json(self.safe_path(discovery["client_profile"]).read_text())
        if not isinstance(self.client, dict) or not all(isinstance(self.client.get(k), str) and self.client[k]
                                                     for k in ("company", "financial_year", "constitution")):
            raise ValueError("Incomplete client metadata")
        index = self.call("list_documents")["documents"]
        self.document_index = {d["doc_id"]: d for d in index}
        if len(self.document_index) != len(index):
            raise ValueError("Duplicate evidence document IDs")
        candidates = []
        for rule_id in discovery["rules"]:
            rule = yaml.safe_load(self.read_document(rule_id))
            if isinstance(rule, dict) and str(rule.get("fy")) == self.client["financial_year"]:
                candidates.append((rule_id, rule))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one rules file matching the engagement financial year")
        self.rules_id, self.rules = candidates[0]
        self.start = date.fromisoformat(str(self.rules["period_from"]))
        self.end = date.fromisoformat(str(self.rules["period_to"]))
        if self.end < self.start:
            raise ValueError("Invalid rule period")
        tables = {table["name"]: table for table in discovery["tables"]}
        if len(tables) != len(discovery["tables"]):
            raise ValueError("Duplicate table definitions")
        self.parties = self.load_csv(tables["party_master"],
                                     {"party", "pan", "entity_type", "city", "gstin", "remarks"})
        ledger = self.load_csv(tables["expense_ledger"],
                               {"date", "voucher_no", "voucher_type", "ledger", "party", "narration", "amount", "tds_section", "tds_amount"})
        self.payees, self.transactions, self.groups = {}, {}, {}
        master = {}
        for record in self.parties:
            name = record["party"]
            if not name.strip() or name in master:
                raise ValueError("Duplicate/empty party identity; conflicting PAN assignments cannot be resolved")
            pan = pan_id(record["pan"])
            identity = f"PAN:{pan}" if pan else f"PARTY:{name}"
            master[name] = identity
            payee = self.payees.setdefault(identity, {"payee_id": identity, "pan": pan, "parties": [], "vouchers": []})
            payee["parties"].append(record)
            expected = {"individual": "P", "huf": "H", "company": "C", "firm": "F", "llp": "F"}.get(record["entity_type"].strip().lower())
            if pan and expected and pan[3] != expected:
                payee.setdefault("identity_issues", []).append(f"Party master entity_type conflicts with PAN for {name}")
        for index, record in enumerate(ledger, 2):
            voucher = record["voucher_no"]
            if not voucher.strip() or voucher in self.transactions:
                raise ValueError(f"Empty/duplicate voucher at CSV row {index}")
            if record["party"] not in master:
                raise ValueError(f"Unmapped party at CSV row {index}; no name-based guess made")
            credited = date.fromisoformat(record["date"])
            if not self.start <= credited <= self.end:
                raise ValueError(f"Voucher {voucher} is outside the rules period")
            identity = master[record["party"]]
            group_key = (identity, record["ledger"])
            if group_key not in self.groups:
                self.groups[group_key] = {"payee_id": identity, "ledger": record["ledger"], "vouchers": []}
            self.groups[group_key]["vouchers"].append(voucher)
            self.transactions[voucher] = {**record, "amount": money(record["amount"]),
                                           "tds_amount": money(record["tds_amount"]),
                                           "source_row": index, "payee_id": identity,
                                           "source": tables["expense_ledger"]["file"],
                                           "raw_amount": record["amount"], "raw_tds_amount": record["tds_amount"]}
            self.payees[identity]["vouchers"].append(voucher)
            self.coverage[voucher] = "unprocessed"
        if not ledger:
            self.limitations.append("Empty expense ledger: financial coverage cannot be certified.")
        if not self.parties:
            self.limitations.append("Empty party master: financial coverage cannot be certified.")
        if output:
            output.trace("ingestion", ledger_rows=len(ledger), party_rows=len(self.parties), payees=len(self.payees),
                         groups=len(self.groups), rules_id=self.rules_id,
                         amount=money_sum(r["amount"] for r in self.transactions.values()),
                         deducted=money_sum(r["tds_amount"] for r in self.transactions.values()))

    def safe_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or ".reviewer" in path.parts or path.name == "answer_key_dev.enc":
            raise ValueError("Source path is outside the public engagement")
        return path

    def call(self, name, **arguments):
        self._calls += 1
        call_id = f"ingestion_{self._calls}"
        if self.output:
            self.output.trace("tool_start", tool_call_id=call_id, name=name, arguments=arguments)
        try:
            result = strict_json(self.kit.functions[name](**arguments))
            if self.output:
                self.output.trace("tool_result", tool_call_id=call_id, name=name, arguments=arguments, result=result)
        except BaseException as exc:
            if self.output:
                self.output.trace("tool_result", tool_call_id=call_id, name=name, arguments=arguments,
                                  result={"success": False, "data": self.output.error(exc)})
            raise
        if not result["success"]:
            raise ValueError(f"Source retrieval failed: {name}: {result.get('error_code')}")
        return result["data"]

    def read_document(self, doc_id):
        if doc_id in self.validated_documents:
            return self.validated_documents[doc_id]
        if doc_id in self.documents:
            return self.documents[doc_id]
        if doc_id not in self.document_index:
            raise ValueError("Unknown evidence document ID")
        total = self.document_index[doc_id]["chars"]
        if type(total) is not int or not 0 <= total <= 1_000_000:
            raise ValueError("Document size is invalid or exceeds the supported 1MB evidence bound")
        chunks, offset = [], 0
        while True:
            result = self.call("read_document", doc_id=doc_id, offset=offset, limit=6000)
            text, next_offset = result["text"], result["next_offset"]
            if result.get("doc_id") != doc_id or result["offset"] != offset or result["total_chars"] != total or not isinstance(text, str):
                raise ValueError("Document changed or has inconsistent pagination metadata")
            end = offset + len(text)
            if end > total or (next_offset is not None and (type(next_offset) is not int or next_offset != end or end <= offset)):
                raise ValueError("Incomplete/non-progressing document retrieval")
            chunks.append(text)
            if next_offset is None:
                if end != total:
                    raise ValueError("Document retrieval terminated before all content was returned")
                break
            offset = next_offset
        self.documents[doc_id] = "".join(chunks)
        return self.documents[doc_id]

    def load_csv(self, table, required):
        with self.safe_path(table["file"]).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if not required <= set(fields) or len(fields) != len(set(fields)):
                raise ValueError(f"Missing/duplicate required CSV fields: {table['name']}")
            rows = list(reader)
        if len(rows) != table["rows"] or any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError("Incomplete or malformed CSV records")
        return rows

    def select(self, vouchers):
        if not vouchers or len(vouchers) != len(set(vouchers)):
            raise ValueError("Provide nonempty unique voucher references")
        if any(v not in self.transactions for v in vouchers):
            raise ValueError("Unknown voucher reference")
        rows = [self.transactions[v] for v in vouchers]
        if len({r["payee_id"] for r in rows}) != 1:
            raise ValueError("Financial aggregation cannot cross normalized PAN/payee boundaries")
        return sorted(rows, key=lambda r: (r["date"], r["source_row"]))

    def coverage_report(self):
        counts = Counter(self.coverage.values())
        groups = [{"ledger": g["ledger"], "payee_id": g["payee_id"],
                   "states": dict(Counter(self.coverage[v] for v in g["vouchers"]))} for g in self.groups.values()]
        return {"transactions": len(self.transactions), "states": dict(counts), "groups": groups,
                "payees": len(self.payees), "complete": bool(self.transactions) and not self.limitations and
                all(s in ("assessed", "compliant") for s in self.coverage.values()),
                "unresolved": counts["unresolved"], "unprocessed": counts["unprocessed"]}

    def get_party_summary(self, payee_id=None, offset=0, limit=20):
        if payee_id is not None and payee_id not in self.payees:
            raise ValueError("Unknown payee identity")
        identities = [payee_id] if payee_id else list(self.payees)
        items = []
        for identity in identities:
            payee = self.payees[identity]
            rows = [self.transactions[v] for v in payee["vouchers"]]
            items.append({"payee_id": identity, "pan": payee["pan"], "parties": payee["parties"],
                          "identity_issues": payee.get("identity_issues", []),
                          "rows": len(rows), "amount": money_sum(r["amount"] for r in rows),
                          "tds_deducted": money_sum(r["tds_amount"] for r in rows),
                          "ledgers": sorted({r["ledger"] for r in rows}),
                          "states": dict(Counter(self.coverage[v] for v in payee["vouchers"]))})
        result = page(items, offset, limit)
        result["coverage"] = {k: v for k, v in self.coverage_report().items() if k != "groups"}
        result["rules_id"] = self.rules_id
        return result

    def get_party_transactions(self, payee_id, offset=0, limit=20):
        if payee_id not in self.payees:
            raise ValueError("Unknown payee identity")
        return page([self.transactions[v] for v in self.payees[payee_id]["vouchers"]], offset, limit)
