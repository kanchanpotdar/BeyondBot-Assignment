"""Synthetic public engagement fixtures; never use protected assessment data."""
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from toolkit import Toolkit
from tds_agent.data import Engagement
from tds_agent.output import Output, exact_json

ROOT = Path(__file__).resolve().parents[1]
PAN = "AAAFA0001A"
PERSON_PAN = "AAAPA0001A"
OTHER_PAN = "AAAFA0002A"
CLIENT = {"company": "Synthetic Buyer Limited", "financial_year": "2025-26",
          "constitution": "Private limited company (not a bank)", "tan": "PNEB00001A"}
LEDGER_FIELDS = "date voucher_no voucher_type ledger party narration amount tds_section tds_amount".split()
PARTY_FIELDS = "party pan entity_type city gstin remarks".split()


def row(voucher="V1", amount="40000", party="Vendor", ledger="Services", date="2025-04-01", tds="0", **extra):
    return {"date": date, "voucher_no": voucher, "voucher_type": "Journal", "ledger": ledger,
            "party": party, "narration": "Synthetic service payment", "amount": amount,
            "tds_section": "", "tds_amount": tds, **extra}


def party(name="Vendor", pan=PAN, entity_type="Firm"):
    return dict(zip(PARTY_FIELDS, [name, pan, entity_type, "Test City", "", ""]))


def cert(pan=PAN, tan="PNEB00001A", rate="1.00", cap="60000", start="01-10-2025", end="31-03-2026", section="194J — professional services"):
    return f"""CERTIFICATE UNDER SECTION 197
Deductee: Synthetic Vendor (PAN {pan})
Deductor: Synthetic Buyer Limited (TAN {tan})
Section: {section}
Rate of deduction: {rate} per cent
Valid from: {start}
Valid to: {end}
Amount up to which this certificate is valid: Rs {cap}
"""


def transporter(pan=PAN, maximum="ten", count=6, fy="2025-26", company=CLIENT["company"]):
    return f"""DECLARATION UNDER SECTION 194C(6)
To: {company}
I (PAN {pan}) am engaged in the business of plying, hiring and leasing goods carriages.
I declare that I do not own more than {maximum} goods carriages at any time during the previous year {fy}.
Total: {count} goods carriages.
"""


def turnover(amount="100000001", fy="2024-25", company=CLIENT["company"]):
    return f"Company: {company}\nFinancial year: {fy}\nBusiness turnover: Rs {amount}\n"


def fixture(test, rows=None, parties=None, documents=None, rule_change=None, load=True):
    temp = tempfile.TemporaryDirectory()
    test.addCleanup(temp.cleanup)
    root = Path(temp.name)
    for directory in ("books", "rules", "documents"):
        (root / directory).mkdir()
    (root / "client.json").write_text(exact_json(CLIENT))
    rules = yaml.safe_load((ROOT / "data/rules/tds_fy_2025_26.yaml").read_text())
    if rule_change:
        rule_change(rules)
    (root / "rules/engagement_rules.yaml").write_text(yaml.safe_dump(rules, sort_keys=False))
    for name, records, fields in (("expense_ledger", rows if rows is not None else [row()], LEDGER_FIELDS),
                                  ("party_master", parties if parties is not None else [party()], PARTY_FIELDS)):
        with (root / "books" / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    for doc_id, text in (documents or {}).items():
        (root / "documents" / (doc_id + ".txt")).write_text(text)
    # Synthetic engagements intentionally have no private reviewer backing files.
    # Keep real public Toolkit data methods; isolate only reviewer construction.
    with patch("toolkit.tools.Reviewer"):
        kit = Toolkit(root)
    output = Output(root / "output")
    output.attach_schema(ROOT / "findings.schema.json")
    data = Engagement(root, kit, output) if load else None
    return root, kit, output, data
