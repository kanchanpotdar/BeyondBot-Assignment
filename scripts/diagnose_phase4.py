"""Offline call attribution from public inputs and preserved traces only."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def diagnose(folder, ledger):
    records = [json.loads(line) for line in (folder / 'trace.jsonl').read_text().splitlines()]
    calculations = {r['calculation']['calculation_id']: r['calculation'] for r in records if r['event'] == 'financial_calculation'}
    steps, current = defaultdict(list), None
    for r in records:
        if r['event'] == 'model_attempt': current = r['attempt']
        elif current is not None: steps[current].append(r)
    phases, payees, documents, errors = Counter(), defaultdict(set), Counter(), Counter()
    reviewer, resolved, zero, progress = [], set(), 0, []
    for attempt, events in steps.items():
        names, new = set(), set()
        for r in events:
            if r['event'] == 'tool_start' and not r['tool_call_id'].startswith('ingestion_'):
                names.add(r['name'])
                try:
                    args = json.loads(r['arguments']) if isinstance(r['arguments'], str) else (r['arguments'] or {})
                except ValueError:
                    args = {}
                identity = args.get('payee_id') or calculations.get(args.get('calculation_id'), {}).get('payee_id')
                if identity: payees[identity].add(attempt)
                if r['name'] == 'assess_batch':
                    for entry in args.get('assessments', []):
                        if isinstance(entry, dict) and entry.get('payee_id'):
                            payees[entry['payee_id']].add(attempt)
                if r['name'] == 'read_document': documents[(args.get('doc_id'), args.get('offset', 0), args.get('limit', 6000))] += 1
            if r['event'] == 'tool_result' and not r['result'].get('success', False): errors[r['result'].get('error_code', 'unspecified')] += 1
            if r['event'] == 'tool_result' and r['name'] == 'ask_reviewer':
                result = r['result']; args = r.get('arguments') or {}; data = result.get('data')
                reviewer.append({'attempt': attempt, 'call_id': r['tool_call_id'], 'fact': args.get('fact'),
                                 'payee_id': args.get('payee_id'), 'success': result.get('success'),
                                 'error_code': result.get('error_code'),
                                 'original_evidence_id': data.get('evidence_id') if isinstance(data, dict) else None})
            if r['event'] == 'assessment_recorded' and r['assessment']['status'] != 'needs_review': new.update(r['assessment']['vouchers'])
        increase = len(new - resolved)
        zero += not bool(increase)
        resolved.update(new)
        progress.append({'attempt': attempt, 'resolved_increase': increase, 'tools': sorted(names)})
        for label, tools in {'discovery': {'list_tables', 'list_documents', 'describe_table'},
                             'retrieval': {'get_party_summary', 'get_party_transactions', 'get_investigation_batch', 'read_document', 'search_documents'},
                             'calculation': {'calculate_tds', 'assess_batch'}, 'recording': {'record_assessment', 'assess_batch'},
                             'reviewer_or_fact_registration': {'ask_reviewer', 'record_evidence'}}.items():
            if names & tools: phases[label] += 1
        if not names: phases['no_tools'] += 1
    final = next(r for r in reversed(records) if r['event'] == 'coverage_final')
    pending_heads = Counter(ledger[v]['ledger'] for v in final['unprocessed_vouchers'])
    return {'calls': len(steps), 'calls_without_resolved_increase': zero, 'phase_calls_nonexclusive': dict(phases),
            'top_payee_call_touches_nonexclusive': sorted(({'payee_id': k, 'calls': len(v)} for k, v in payees.items()), key=lambda x: -x['calls'])[:10],
            'repeated_document_pages': [{'doc_id': k[0], 'offset': k[1], 'limit': k[2], 'requests': v} for k, v in documents.items() if v > 1],
            'tool_errors': dict(errors), 'context_checkpoints': sum(r['event'] == 'context_checkpoint' for r in records),
            'tool_delivery_limits': sum(r['event'] == 'tool_delivery' for r in records),
            'reviewer_calls': reviewer, 'per_call_progress': progress,
            'unprocessed_ledger_heads': dict(pending_heads), 'scorer_summary': json.loads((folder / 'metrics.json').read_text()).get('scoring_result'),
            'scope': 'Call labels/touches overlap; no claim that each zero-coverage call was useless. Exact missed answer-key identities are unavailable from summary feedback.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--cohort', type=Path, default=Path('experiments/controller_recovery'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with (args.data / 'books/expense_ledger.csv').open(newline='', encoding='utf-8-sig') as handle:
        ledger = {r['voucher_no']: r for r in csv.DictReader(handle)}
    result = {p.parent.name: diagnose(p.parent, ledger) for p in sorted(args.cohort.glob('*/metrics.json'))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle: json.dump(result, handle, indent=2)
    print(json.dumps({name: {'calls': r['calls'], 'calls_without_resolved_increase': r['calls_without_resolved_increase']} for name, r in result.items()}))
