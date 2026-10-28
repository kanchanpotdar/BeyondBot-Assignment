"""Read-only ledger/arithmetic/coverage audit; does not certify classifications."""
import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def read_json(text):
    return json.loads(text, parse_float=Decimal)


def audit(data, folder):
    with (Path(data) / "books/expense_ledger.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ledger = {r["voucher_no"]: r for r in rows}
    assert len(ledger) == len(rows), "Duplicate source vouchers"
    folder = Path(folder)
    output = read_json((folder / "findings.json").read_text())
    records = [read_json(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    records = [r for r in records if r["run_id"] == output["run"]["run_id"]]
    calculations = [r["calculation"] for r in records if r["event"] == "financial_calculation"]
    assessments, recorded_ids, replays = [], {}, 0
    calculated_ids = {c['calculation_id']: c for c in calculations if 'calculation_id' in c}
    for record in records:
        if record['event'] != 'assessment_recorded':
            continue
        assessment = record['assessment']
        identity = assessment.get('calculation_id')
        if identity is not None:
            assert identity in calculated_ids, 'Recorded assessment has no stored calculation'
            assert record.get('calculation_id', identity) == identity
            core = {k: v for k, v in assessment.items() if k != 'reason'}
            assert core == {k: v for k, v in calculated_ids[identity].items() if k != 'reason'}, 'Assessment differs from stored calculation'
            if identity in recorded_ids:
                assert core == recorded_ids[identity], 'Replayed calculation changed financial values'
                replays += 1
                continue  # Same calculation replay, not a second voucher assessment.
            recorded_ids[identity] = core
        assessments.append(assessment)
    for c in calculations:
        vouchers = c["vouchers"]
        assert len(vouchers) == len(set(vouchers)) and set(vouchers) <= ledger.keys()
        assert c["amount_total"] == sum((Decimal(ledger[v]["amount"]) for v in vouchers), Decimal(0))
        assert c["tds_deducted"] == sum((Decimal(ledger[v]["tds_amount"]) for v in vouchers), Decimal(0))
        if c["status"] == "needs_review":
            assert c["shortfall"] == 0 and c["needs"]
            continue
        if c['status'] == 'excluded':
            assert c['section'] == c['category'] == 'excluded'
            assert c['amount_liable'] == c['tds_required'] == c['shortfall'] == 0
            assert c['breakdown'] == []
            continue  # Existing calculator contract: no taxable segments.
        assert sorted(b["voucher_no"] for b in c["breakdown"]) == sorted(vouchers)
        for b in c["breakdown"]:
            original = ledger[b["voucher_no"]]
            assert b["amount"] == Decimal(original["amount"])
            assert b["tds_deducted"] == Decimal(original["tds_amount"])
            assert b["date"] == original["date"]
            segments = b.get("rates", [])
            assert b["amount_liable"] == sum((s["base"] for s in segments), Decimal(0))
            assert b["tds_required_exact"] == sum((Decimal(s["base"]) * Decimal(s["rate"]) / 100 for s in segments), Decimal(0))
        required = sum((b["tds_required_exact"] for b in c["breakdown"]), Decimal(0)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
        assert c["tds_required"] == required
        assert c["amount_liable"] == sum((b["amount_liable"] for b in c["breakdown"]), Decimal(0))
        assert c["shortfall"] == max(Decimal(0), required - c["tds_deducted"])
    recorded = [v for a in assessments for v in a["vouchers"]]
    assert len(recorded) == len(set(recorded)), "Vouchers assessed twice"
    final = next(r for r in reversed(records) if r["event"] == "coverage_final")
    unprocessed = final["unprocessed_vouchers"]
    unresolved = final["unresolved_vouchers"]
    assert len(unprocessed) == len(set(unprocessed))
    assert set(recorded).isdisjoint(unprocessed)
    assert set(recorded) | set(unprocessed) == ledger.keys(), "Coverage does not partition all source vouchers"
    assert set(unresolved) == {v for a in assessments if a["status"] == "needs_review" for v in a["vouchers"]}
    for finding in output["findings"]:
        matches = [a for a in assessments if a["party"] == finding["party"] and a["section"] == finding["section"]
                   and sorted(a["vouchers"]) == sorted(finding["vouchers"]) and a["status"] == finding["status"]]
        assert len(matches) == 1, "Finding lacks a unique recorded assessment"
        a = matches[0]
        for field in ("shortfall", "tds_required", "tds_deducted", "amount_liable"):
            if field in finding:
                assert finding[field] == a[field], "Finding amount differs from recorded calculation"
    return {"ledger_rows": len(ledger), "calculations_checked": len(calculations),
            'replayed_assessment_events': replays,
            "recorded_vouchers": len(recorded), "unprocessed": len(unprocessed), "unresolved": len(unresolved),
            "findings_checked": len(output["findings"]), "arithmetic_and_partition_valid": True,
            "limitation": "Checks stored arithmetic and attribution; does not certify classification, evidence sufficiency, or full financial coverage."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.out), indent=2))
