"""Synthetic scheduling/batching tests: no financial truth from a real model."""
import json
import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from phase2_support import fixture, row, party, PAN, OTHER_PAN
from test_phase1 import ScriptLLM, payload, call
from test_phase3 import plan
from tds_agent.agent import Agent, Limits
from tds_agent.output import exact_json, strict_json
from tds_agent.tools import FinancialTools, ToolDispatcher


class SchedulingTests(unittest.TestCase):
    def setup_case(self, **kwargs):
        root, kit, out, data = fixture(self, **kwargs)
        financial = FinancialTools(data, out, investigate=True, scheduled=True)
        dispatch = ToolDispatcher(kit, out, financial=financial)
        return root, kit, out, data, financial, dispatch

    def execute(self, dispatch, name, args, ident='phase5'):
        return strict_json(dispatch.execute(ident, name, exact_json(args)))

    def entry(self, identity='PAN:' + PAN, classifications=None):
        return {'payee_id': identity, 'classifications': classifications or [plan()],
                'reason': 'The quoted source describes the selected synthetic service work.'}

    def test_material_services_outrank_large_low_rate_goods(self):
        _, _, _, data, f, _ = self.setup_case(rows=[
            row('G', '5100000', party='Cargo Inputs', ledger='Raw Material Purchases', narration='Purchase goods'),
            row('S', '200000', party='Zeta Advisors', ledger='Professional Fees', narration='Professional advice')],
            parties=[party('Cargo Inputs'), party('Zeta Advisors', OTHER_PAN)])
        self.assertEqual(f.scheduler.pending()[0], 'PAN:' + OTHER_PAN)
        self.assertEqual(set(f.scheduler.pending()), set(data.payees))
        self.assertTrue(all(v == 'unprocessed' for v in data.coverage.values()))

    def test_rank_uses_yaml_values_not_embedded_rates(self):
        _, _, _, _, f, _ = self.setup_case(rows=[row(amount='60000', narration='Professional fee')])
        f.data.rules['sections']['194J']['rate_others'] = '3'
        self.assertEqual(f.scheduler.screen('PAN:' + PAN)['screening_upper_estimate'], Decimal('1800'))

    def test_exact_invoice_templates_keep_all_references(self):
        _, _, _, data, f, _ = self.setup_case(rows=[
            row('A', narration='Purchase goods — invoice INV/001', ledger='Goods'),
            row('B', narration='Purchase goods — invoice INV/002', ledger='Goods')])
        item = f.scheduler.batch()['items'][0]
        self.assertEqual(item['total_groups'], 1)
        self.assertEqual(item['nature_groups'][0]['vouchers'], ['A', 'B'])
        self.assertEqual(item['nature_groups'][0]['amount'], Decimal('80000'))
        self.assertEqual(f.investigation.observed, set(data.transactions))
        self.assertEqual(data.coverage['A'], 'unprocessed')

    def test_operator_and_financial_conditions_not_normalized_away(self):
        _, _, _, _, f, _ = self.setup_case(rows=[
            row('A', narration='Crane hire with operator'), row('B', narration='Crane hire equipment only')])
        self.assertEqual(f.scheduler.batch()['items'][0]['total_groups'], 2)

    def test_group_pagination_is_explicit_and_does_not_deliver_omitted_rows(self):
        _, _, _, data, f, _ = self.setup_case(rows=[row(str(i), narration=f'Distinct service type {i}') for i in range(10)])
        page = f.scheduler.batch()['items'][0]
        self.assertEqual(page['next_group_offset'], 8)
        self.assertEqual(len(f.investigation.observed), 8)
        self.assertEqual(f.scheduler.batch(payee_id='PAN:' + PAN, offset=8)['items'][0]['next_group_offset'], None)
        self.assertEqual(f.investigation.observed, set(data.transactions))
        self.assertEqual(set(data.coverage.values()), {'unprocessed'})

    def test_batch_records_multiple_payees_and_categories(self):
        _, _, out, data, f, dispatch = self.setup_case(rows=[
            row('V1', amount='60000', ledger='Professional', narration='Professional audit services'),
            row('V2', amount='60000', ledger='Technical', narration='Technical services'),
            row('V3', amount='25000', party='Other', ledger='Commission', narration='Sales commission')],
            parties=[party(), party('Other', OTHER_PAN)])
        f.scheduler.batch(limit=5)
        entries = [self.entry(classifications=[plan(['V1'], '194J', 'professional', 'Professional audit services'),
                                              plan(['V2'], '194J', 'technical', 'Technical services')]),
                   self.entry('PAN:' + OTHER_PAN, [plan(['V3'], '194H', 'commission', 'Sales commission')])]
        result = self.execute(dispatch, 'assess_batch', {'assessments': entries})
        self.assertTrue(all(e['success'] for e in result['data']['results']))
        self.assertEqual(sum(x['tds_required'] for x in out.state['findings']), Decimal('7700'))
        self.assertEqual(len(out.state['findings']), 3)
        self.assertTrue(data.coverage_report()['complete'])

    def test_bad_entry_and_unknown_payee_do_not_discard_valid_entry(self):
        _, _, out, data, f, dispatch = self.setup_case()
        f.scheduler.batch()
        response = self.execute(dispatch, 'assess_batch', {'assessments': [self.entry(), {'payee_id': 'invented'}, {'payee_id': 'PAN:' + PAN, 'rate': 0}]})
        self.assertEqual([e['success'] for e in response['data']['results']], [True, False, False])
        self.assertEqual(out.state['findings'][0]['shortfall'], Decimal('800'))
        self.assertEqual(data.coverage['V1'], 'assessed')

    def test_invalid_classification_keeps_coverage_unprocessed(self):
        _, _, _, data, f, dispatch = self.setup_case()
        f.scheduler.batch()
        bad = self.entry(classifications=[plan(category='invented_category')])
        result = self.execute(dispatch, 'assess_batch', {'assessments': [bad]})
        self.assertFalse(result['data']['results'][0]['success'])
        self.assertEqual(data.coverage['V1'], 'unprocessed')

    def test_deferral_preserves_queue_and_does_not_emit_financial_finding(self):
        _, _, out, data, f, dispatch = self.setup_case()
        result = self.execute(dispatch, 'assess_batch', {'assessments': [{'payee_id': 'PAN:' + PAN, 'defer_reason': 'Need service agreement'}]})
        self.assertTrue(result['data']['results'][0]['deferred'])
        self.assertEqual(f.scheduler.state('PAN:' + PAN), 'awaiting_evidence')
        self.assertEqual(f.scheduler.pending(), ['PAN:' + PAN])
        self.assertEqual(data.coverage['V1'], 'unprocessed')
        self.assertEqual(out.state['findings'], [])

    def test_source_caching_and_shared_fact_alias_preserve_original_id(self):
        _, kit, _, _, f, dispatch = self.setup_case(rows=[row(amount='5100000', ledger='Goods', narration='Purchase goods')])
        f.investigation.search('buyer turnover')
        kit.reviewer.ask.return_value = {'status': 'ANSWERED', 'answer': 'Synthetic Buyer Limited turnover in FY 2024-25 was Rs 12 crore.', 'questions_left': 2}
        args = {'question': 'Did Synthetic Buyer Limited exceed the prior-year turnover condition?', 'why': 'Buyer eligibility affects material goods liability.', 'fact': 'buyer_turnover', 'payee_id': 'Synthetic Buyer Limited'}
        a = self.execute(dispatch, 'ask_reviewer', args, 'original')
        b = self.execute(dispatch, 'ask_reviewer', {**args, 'payee_id': 'PAN:' + OTHER_PAN, 'question': 'Confirm the buyer turnover please.'}, 'alias')
        self.assertEqual(a['data']['evidence_id'], 'reviewer:original')
        self.assertEqual(a['data']['evidence_id'], b['data']['evidence_id'])
        kit.reviewer.ask.assert_called_once()
        f.investigation.overview()
        with patch.dict(kit.functions, {'read_document': lambda **_: self.fail('Cached source must not be refetched')}):
            f.investigation.document_page(f.data.rules_id)

    def test_shared_buyer_question_prioritized_before_low_impact_reimbursement(self):
        _, kit, _, _, f, dispatch = self.setup_case(rows=[
            row('G', amount='6000000', ledger='Goods', narration='Purchase goods'),
            row('E', amount='200', party='Employee', ledger='Travel', narration='Train reimbursement')],
            parties=[party(), party('Employee', OTHER_PAN)])
        f.investigation.search('Employee receipts')
        response = self.execute(dispatch, 'ask_reviewer', {'payee_id': 'PAN:' + OTHER_PAN, 'fact': 'payment_nature',
            'question': 'Are Employee payments exact reimbursements?', 'why': 'Determine financial nature of this payment.'})
        self.assertFalse(response['success'])
        self.assertIn('shared buyer', response['data']['message'])
        kit.reviewer.ask.assert_not_called()

    def test_generic_consultancy_requires_real_resolution(self):
        _, _, _, data, f, dispatch = self.setup_case(rows=[row(amount='60000', narration='Consultancy — plant layout optimisation')])
        f.scheduler.batch()
        result = self.execute(dispatch, 'assess_batch', {'assessments': [self.entry(classifications=[
            plan(section='194J', category='professional', quote='Consultancy — plant layout optimisation')])]})
        self.assertFalse(result['data']['results'][0]['success'])
        self.assertEqual(data.coverage['V1'], 'unprocessed')

    def test_partial_checkpoint_failure_preserves_earlier_payee(self):
        _, _, out, data, f, dispatch = self.setup_case(rows=[row(), row('V2', party='Other')], parties=[party(), party('Other', OTHER_PAN)])
        f.scheduler.batch()
        checkpoint = out.checkpoint
        def flaky(candidate):
            if len(candidate['findings']) > 1: raise OSError('Synthetic disk failure')
            return checkpoint(candidate)
        with patch.object(out, 'checkpoint', side_effect=flaky):
            result = self.execute(dispatch, 'assess_batch', {'assessments': [self.entry(), self.entry('PAN:' + OTHER_PAN, [plan(['V2'])])]})
        self.assertEqual([e['success'] for e in result['data']['results']], [True, False])
        self.assertEqual(len(out.state['findings']), 1)
        self.assertEqual(data.coverage['V2'], 'unprocessed')

    def test_two_turn_native_batch_review_generalizes_to_second_client(self):
        _, _, out, data, f, dispatch = self.setup_case(rows=[row('A', party='Harbor Engineering'), row('B', party='Cedar Works')],
            parties=[party('Harbor Engineering'), party('Cedar Works', OTHER_PAN)])
        data.client['company'] = 'Independent Synthetic Buyer Two'
        entries = [self.entry('PAN:' + PAN, [plan(['A'])]), self.entry('PAN:' + OTHER_PAN, [plan(['B'])])]
        model = ScriptLLM([payload(calls=[call('assess_batch', exact_json({'assessments': entries}), ident='batch-id')]), payload('{"findings":[]}')])
        agent = Agent(model, dispatch, out, Limits(max_calls=2, max_tools=5))
        self.assertEqual(agent.run(), 'financial_review_complete')
        self.assertEqual(agent.attempts, 2)
        self.assertTrue(data.coverage_report()['complete'])
        self.assertEqual(sum(x['tds_required'] for x in out.state['findings']), Decimal('1600'))
        self.assertTrue(any(m.get('tool_call_id') == 'batch-id' for m in agent.messages))

    def test_unproductive_calls_reprioritize_then_stop(self):
        _, _, out, data, _, dispatch = self.setup_case()
        replies = [payload(calls=[call('list_documents', ident=f'noprogress{i}')]) for i in range(10)]
        agent = Agent(ScriptLLM(replies), dispatch, out, Limits(max_calls=15, max_tools=20))
        self.assertEqual(agent.run(), 'no_financial_progress')
        self.assertEqual(agent.attempts, 9)
        self.assertEqual(data.coverage['V1'], 'unprocessed')

    def test_missing_cost_and_call_limits_preserve_batch_checkpoints(self):
        _, _, out, data, f, dispatch = self.setup_case()
        response = payload(calls=[call('assess_batch', exact_json({'assessments': [self.entry()]}))])
        response.pop('usage', None)
        agent = Agent(ScriptLLM([response]), dispatch, out, Limits(max_calls=3, require_cost_metadata=True))
        self.assertEqual(agent.run(), 'cost_metadata_unavailable')
        self.assertEqual(data.coverage['V1'], 'assessed')
        self.assertEqual(len(out.state['findings']), 1)

    def test_supplied_fake_server_native_batch_with_real_wrapper_and_trace(self):
        from toolkit import LLM
        from toolkit.fake_llm import FakeLLM
        _, _, out, data, _, dispatch = self.setup_case()
        script = {'steps': [{'tool_calls': [{'name': 'assess_batch', 'arguments': exact_json({'assessments': [self.entry()]})}]},
                            {'content': '{"findings":[]}'}]}
        server = FakeLLM(script).start()
        self.addCleanup(server.stop)
        with patch.dict(os.environ, {'LLM_BASE_URL': server.url, 'LLM_MODEL': 'local-phase5', 'LLM_API_KEY': 'fake',
                                     'NO_PROXY': '127.0.0.1,localhost'}):
            agent = Agent(LLM(timeout=3), dispatch, out, Limits(max_calls=3, max_tools=5))
            reason = agent.run()
        self.assertEqual(reason, 'financial_review_complete')
        out.finalize(reason, agent.metrics())
        self.assertEqual(agent.attempts, 2)
        self.assertEqual(len(server.requests[1]['tool_results']), 1)
        self.assertTrue(data.coverage_report()['complete'])
        records = [strict_json(line) for line in out.trace_path.read_text().splitlines()]
        result = next(r for r in records if r['event'] == 'tool_result' and r['name'] == 'assess_batch')
        start = next(r for r in records if r['event'] == 'tool_start' and r['name'] == 'assess_batch')
        self.assertEqual(start['tool_call_id'], result['tool_call_id'])
        self.assertEqual(result['result']['data']['results'][0]['calculations'][0]['tds_required'], Decimal('800'))

    def test_low_risk_unvisited_groups_do_not_crowd_out_recovered_material_work(self):
        _, _, _, _, f, _ = self.setup_case(rows=[row('H', '200000', narration='Professional advice'),
            row('L', '100', party='Other', narration='Travel reimbursement')], parties=[party(), party('Other', OTHER_PAN)])
        f.scheduler.batch(limit=1)
        f.scheduler.rotate()
        self.assertEqual(f.scheduler.pending()[0], 'PAN:' + OTHER_PAN)  # Short cooldown.
        f.scheduler.tick += 3
        self.assertEqual(f.scheduler.pending()[0], 'PAN:' + PAN)

    def test_real_new_evidence_pages_are_not_treated_as_stalled_work(self):
        _, _, out, data, _, dispatch = self.setup_case(documents={'agreement': 'Evidence sentence. ' * 1000})
        replies = [payload(calls=[call('read_document', exact_json({'doc_id': 'agreement', 'offset': i * 1000, 'limit': 1000}), ident=f'page{i}')]) for i in range(10)]
        agent = Agent(ScriptLLM(replies), dispatch, out, Limits(max_calls=10, max_tools=20))
        self.assertEqual(agent.run(), 'model_call_limit')
        self.assertEqual(agent.attempts, 10)
        self.assertEqual(data.coverage['V1'], 'unprocessed')

    def test_repeated_batch_does_not_emit_duplicate_assessment_events(self):
        _, _, out, _, f, dispatch = self.setup_case()
        f.scheduler.batch()
        for i in range(2):
            result = self.execute(dispatch, 'assess_batch', {'assessments': [self.entry()]}, str(i))
            self.assertTrue(result['data']['results'][0]['success'])
        records = [strict_json(line) for line in out.trace_path.read_text().splitlines()]
        self.assertEqual(sum(r['event'] == 'assessment_recorded' for r in records), 1)

    def test_reported_cost_limit_preserves_completed_batch(self):
        _, _, out, data, _, dispatch = self.setup_case()
        response = payload(calls=[call('assess_batch', exact_json({'assessments': [self.entry()]}))])
        response['usage']['cost'] = 0.01
        agent = Agent(ScriptLLM([response]), dispatch, out, Limits(max_calls=3, max_cost_usd=0.005))
        self.assertEqual(agent.run(), 'cost_limit')
        self.assertEqual(data.coverage['V1'], 'assessed')
        self.assertEqual(len(out.state['findings']), 1)

    def test_genuine_unknown_is_recorded_without_inventing_liability(self):
        _, _, out, data, f, dispatch = self.setup_case(rows=[row(narration='Consultancy — unresolved payment nature')])
        f.scheduler.batch()
        entry = self.entry(classifications=[plan(section='unknown', category='unknown', quote='Consultancy — unresolved payment nature',
                                               needs='Need the actual professional or technical scope of the consultancy.')])
        result = self.execute(dispatch, 'assess_batch', {'assessments': [entry]})
        self.assertTrue(result['data']['results'][0]['success'])
        self.assertEqual(data.coverage['V1'], 'unresolved')
        self.assertEqual(out.state['findings'][0]['shortfall'], 0)
        self.assertEqual(f.scheduler.state('PAN:' + PAN), 'unresolved')

    def test_global_fact_priority_does_not_deadlock_material_scoped_questions(self):
        _, kit, _, _, f, dispatch = self.setup_case(rows=[row('G', '6000000', ledger='Goods', narration='Purchase goods'),
            row('S', '200000', party='Other', narration='Professional consultancy')], parties=[party(), party('Other', OTHER_PAN)])
        f.investigation.search('Other agreement')
        kit.reviewer.ask.return_value = {'status': 'UNAVAILABLE', 'answer': 'No additional information.', 'questions_left': 2}
        args = {'fact': 'payment_nature', 'payee_id': 'PAN:' + OTHER_PAN, 'question': 'What was the actual nature of Other consultancy?',
                'why': 'The service category materially changes the applicable TDS rate.'}
        result = self.execute(dispatch, 'ask_reviewer', args)
        self.assertTrue(result['success'])
        self.assertIn('shared_fact_task', f.scheduler.summary())
        kit.reviewer.ask.assert_called_once()

    def test_last_question_slot_reserved_for_unanswered_shared_material_fact(self):
        _, kit, _, _, f, dispatch = self.setup_case(rows=[row('G', '6000000', ledger='Goods', narration='Purchase goods'),
            row('S', '200000', party='Other', narration='Professional consultancy')], parties=[party(), party('Other', OTHER_PAN)])
        f.investigation.search('Other agreement')
        kit.reviewer.ask.return_value = {'status': 'UNAVAILABLE', 'answer': 'No additional information.', 'questions_left': 1}
        args = {'payee_id': 'PAN:' + OTHER_PAN, 'question': 'Please clarify Other material facts.', 'why': 'Missing facts affect the actual service liability.'}
        for i, fact in enumerate(('payment_nature', 'identity')):
            self.assertTrue(self.execute(dispatch, 'ask_reviewer', {**args, 'fact': fact}, str(i))['success'])
        result = self.execute(dispatch, 'ask_reviewer', {**args, 'fact': 'transporter_eligibility'}, 'reserved')
        self.assertFalse(result['success'])
        self.assertIn('reserved', result['data']['message'])
        self.assertEqual(kit.reviewer.ask.call_count, 2)

    def test_verified_employee_role_is_delivered_and_cited_exclusion_retains_coverage(self):
        person = {**party('Staff Member', '', 'Individual'), 'remarks': 'Employee expense reimbursements'}
        _, _, out, data, f, dispatch = self.setup_case(rows=[row(party='Staff Member', amount='4000', ledger='Travel',
            narration='Employee expense reimbursement - train fare')], parties=[person])
        batch = f.scheduler.batch()['items'][0]
        self.assertEqual(batch['entity_details'][0]['remarks'], 'Employee expense reimbursements')
        entry = self.entry('PARTY:Staff Member', [plan(section='excluded', category='excluded',
            quote='Employee expense reimbursement - train fare')])
        self.assertTrue(self.execute(dispatch, 'assess_batch', {'assessments': [entry]})['data']['results'][0]['success'])
        self.assertEqual(out.state['findings'], [])
        self.assertEqual(data.coverage['V1'], 'compliant')
        self.assertTrue(data.coverage_report()['complete'])

    def test_legacy_recording_tool_retries_are_idempotent_too(self):
        _, _, out, data, f, dispatch = self.setup_case()
        f.scheduler.batch()
        calculation = f.calculate_tds('PAN:' + PAN, [plan()])['calculations'][0]['calculation_id']
        for i in range(2):
            result = self.execute(dispatch, 'record_assessment', {'calculation_id': calculation,
                                  'reason': 'The synthetic source supports contractor work.'}, 'record' + str(i))
            self.assertTrue(result['success'])
        self.assertTrue(result['data']['already_recorded'])
        records = [strict_json(line) for line in out.trace_path.read_text().splitlines()]
        self.assertEqual(sum(r['event'] == 'assessment_recorded' for r in records), 1)
        self.assertEqual(data.coverage['V1'], 'assessed')
