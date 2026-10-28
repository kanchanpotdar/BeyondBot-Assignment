# BeyondBot TDS Review Agent

A Python CLI agent for investigating tax deducted at source (TDS) exceptions. It combines model-assisted evidence classification with deterministic, auditable financial calculations.

## What it does

- Loads the full ledger and party master, groups records by normalized PAN or party identity, and tracks voucher coverage.
- Uses a bounded native tool-calling loop for classification and selective document investigation.
- Applies YAML-driven TDS rules with Decimal arithmetic; the model does not supply transaction amounts or rates.
- Validates source references, records assessment and reviewer evidence, and writes schema-validated findings plus an append-only trace.
- Preserves partial results and clearly reports unresolved or unprocessed coverage.

## Architecture

`run.py` starts the CLI and coordinates `tds_agent/agent.py`, `investigation.py`, `tools.py`, `data.py`, `calculations.py`, `scheduler.py`, and `output.py`. The supplied Toolkit provides the LLM wrapper, source-document and ledger tools, and reviewer interface. Experiment orchestration is in `experiments/run_experiments.py`; focused tests are in `tests/`.

## Run

Python 3.10+ is required. Install the declared dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in your environment, then run:

```sh
python run.py --data data --out out
```

This source-only repository intentionally omits the employer-supplied Toolkit, client data, rules, schema, scorer, encoded evaluation assets, real-run outputs, and credentials. The CLI therefore requires the corresponding supplied assessment assets to be restored beside this code before it can run. No `.env`, `.gitignore`, virtual environment, client records, or experiment traces are included here.

To run the included test suite in the assessment environment:

```sh
python -B -m unittest discover -s tests -q
```

The tests also require the supplied Toolkit and assessment fixtures. Use the provided fake model for offline integration checks; do not use real credentials for tests.

## Evaluation snapshot

On the supplied development assessment, the selected confirmation run scored **92.5** and found all 12 scored exceptions with the correct amounts. It resolved 550 of 1,291 vouchers; 738 remained unprocessed and 3 unresolved, so the financial review was incomplete. Earlier successful Phase 4 runs scored 13.4–21.9; Phase 5 runs scored 14.0, 53.9, 73.4, and 92.5. The real-model results are specific to this dataset and do not establish complete coverage or performance on another client.

The offline regression suite passed **149 tests** (0 failures, 0 skipped). The final idempotency guard was verified offline after the selected scored run. Exact experiment traces and client outputs are intentionally excluded from this code-only repository.

## Design limits

This is an assessment prototype, not a tax filing or compliance product. The measured run did not review the full ledger. Evidence interpretation, certificate overlap, reversals and cross-year transactions, rounding assumptions, and recorded-payee reclassification remain limitations. Review the implementation and test it against complete, authorized inputs before relying on it.
