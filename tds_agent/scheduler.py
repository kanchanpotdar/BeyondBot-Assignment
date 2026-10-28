"""Risk-led work queue. Screening estimates never establish tax or compliance."""
from collections import Counter, defaultdict
from decimal import Decimal
import re

from .data import money_sum
from .output import exact_json


class Scheduler:
    def __init__(self, financial):
        self.financial, self.data = financial, financial.data
        self.deferred, self.visits = {}, Counter()
        self.active = []
        self.tick, self.cooldown = 0, {}
        self.entries = {identity: self.screen(identity) for identity in self.data.payees}

    def screen(self, identity):
        payee = self.data.payees[identity]
        rows = [self.data.transactions[v] for v in payee['vouchers']]
        text = ' '.join(r['ledger'] + ' ' + r['narration'] for r in rows).lower()
        candidates = set(r['tds_section'].strip().replace('-', '') for r in rows) & self.data.rules['sections'].keys()
        for pattern, section in ((r'contract|labour|fabricat|job work|security|housekeep|freight|courier|advertis|canteen|repair', '194C'),
                                 (r'professional|technical|consult|audit|legal|medical|architect|software|director|valuation|secretarial', '194J'),
                                 (r'rent|hire|lease', '194I'), (r'commission|brokerage', '194H'),
                                 (r'interest', '194A'), (r'purchase|raw material|goods|packing material', '194Q')):
            if re.search(pattern, text):
                candidates.add(section)
        total = money_sum(r['amount'] for r in rows)
        deducted = money_sum(r['tds_amount'] for r in rows)
        estimates = {}
        for section in candidates:
            rule = self.data.rules['sections'][section]
            base = total
            if section == '194Q':
                base = max(Decimal(0), total - Decimal(rule['threshold_yearly_per_seller']))
                rate = Decimal(rule['rate_on_excess'])
            elif section == '194I':
                months = defaultdict(list)
                for r in rows:
                    months[r['date'][:7]].append(r['amount'])
                base = money_sum(money_sum(v) for v in months.values() if money_sum(v) > Decimal(rule['threshold_monthly']))
                rate = max(Decimal(rule[k]) for k in ('rate_plant_machinery_equipment', 'rate_land_building_furniture'))
            elif section == '194C':
                if total <= Decimal(rule['threshold_yearly']):
                    base = money_sum(r['amount'] for r in rows if r['amount'] > Decimal(rule['threshold_single_payment']))
                rate = Decimal(rule['rate_individual_huf' if payee['pan'] and payee['pan'][3] in 'PH' else 'rate_others'])
            elif section == '194J':
                threshold = rule['threshold_director'] if re.search(r'director|sitting fee', text) else rule['threshold_yearly_per_category']
                base = total if total > Decimal(threshold) else Decimal(0)
                rate = max(Decimal(rule[k]) for k in ('rate_technical_services', 'rate_others'))
            else:
                threshold = rule['threshold_yearly'] if section == '194H' else rule['threshold_yearly_payer_not_bank']
                base = total if total > Decimal(threshold) else Decimal(0)
                rate = Decimal(rule['rate'])
            if not payee['pan']:
                rate = max(rate, Decimal(self.data.rules['no_pan_206AA']['rate_194Q' if section == '194Q' else 'rate']))
            estimates[section] = max(Decimal(0), base * rate / 100 - deducted)
        flags = []
        if not payee['pan']: flags.append('missing_pan')
        if not deducted: flags.append('no_recorded_tds')
        if len(candidates) > 1 or re.search(r'consultancy|miscellaneous|hire', text): flags.append('classification_ambiguity')
        if '194Q' in estimates and estimates['194Q'] > 0: flags.append('shared_buyer_eligibility')
        documents = [doc for doc, meta in self.data.document_index.items()
                     if any(p['party'].casefold() in meta['title'].casefold() for p in payee['parties'])]
        return {'payee_id': identity, 'risk_signals': flags, 'candidate_sections': sorted(candidates),
                'screening_upper_estimate': max(estimates.values(), default=Decimal(0)),
                'goods_eligibility_impact': estimates.get('194Q', Decimal(0)), 'document_links': documents,
                'warning': 'Unverified screening estimate; ignores exceptions/category splits. Never a liability or compliance decision.'}

    def state(self, identity):
        states = {self.data.coverage[v] for v in self.data.payees[identity]['vouchers']}
        if 'unprocessed' not in states:
            if 'unresolved' in states: return 'unresolved'
            return 'non_compliant' if 'assessed' in states else 'compliant_or_excluded'
        if identity in self.financial.plans: return 'ready_for_calculation_or_recording'
        return 'awaiting_evidence' if identity in self.deferred else 'unprocessed'

    def pending(self):
        identities = [i for i in self.entries if 'unprocessed' in {self.data.coverage[v] for v in self.data.payees[i]['vouchers']}]
        return sorted(identities, key=lambda i: (i in self.deferred, self.cooldown.get(i, 0) > self.tick,
                      -self.entries[i]['screening_upper_estimate'], self.visits[i], i))

    def summary(self):
        result = {'states': dict(Counter(self.state(i) for i in self.entries)),
                'active_payees': self.active, 'next_payees': self.pending()[:5],
                'deferred': {k: v[:280] for k, v in list(self.deferred.items())[:5]},
                'deferred_total': len(self.deferred), 'coverage_advances_only_on_recording': True}
        shared = 'buyer_turnover:' + self.data.client['financial_year']
        if shared not in self.financial.investigation.questions and any(e['goods_eligibility_impact'] > 0 for e in self.entries.values()):
            previous = self.data.start.year - 1
            fy = f'{previous}-{str(previous + 1)[-2:]}'
            result['shared_fact_task'] = {'instruction': 'Inspect public turnover sources; ask once if still missing. One reviewer slot is reserved.',
                'tool': 'ask_reviewer', 'suggested_arguments': {'fact': 'buyer_turnover',
                'question': f"For {self.data.client['company']}, did business turnover in FY {fy} exceed Rs {self.data.rules['sections']['194Q']['buyer_turnover_condition']}? Provide a source-backed fact or say unavailable.",
                'why': 'Shared buyer eligibility can change the goods-purchase liability of several sellers.'}}
        return result

    @staticmethod
    def nature_key(row):
        text = row['narration'].casefold()
        # Only explicit invoice references and period tokens are templated. Never
        # strip financial amounts, fleet counts, operator terms or service phases.
        text = re.sub(r'\binvoice\s+[a-z0-9][a-z0-9/._-]*', 'invoice <reference>', text)
        text = re.sub(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', '<month>', text)
        return row['ledger'], re.sub(r'\s+', ' ', text).strip()

    def party_page(self, identity, offset=0):
        groups = defaultdict(list)
        for voucher in self.data.payees[identity]['vouchers']:
            row = self.data.transactions[voucher]
            groups[self.nature_key(row)].append(row)
        all_groups = list(groups.values())
        selected = all_groups[offset:offset + 8]
        result = {**self.entries[identity], 'queue_state': self.state(identity),
                  'parties': [p['party'] for p in self.data.payees[identity]['parties']],
                  'entity_details': [{k: p[k] for k in ('party', 'entity_type', 'remarks')}
                                     for p in self.data.payees[identity]['parties']],
                  'group_offset': offset, 'total_groups': len(all_groups), 'nature_groups': []}
        for rows in selected:
            result['nature_groups'].append({'ledger': rows[0]['ledger'], 'source_voucher': rows[0]['voucher_no'],
                'quote': rows[0]['narration'], 'vouchers': [r['voucher_no'] for r in rows],
                'amount': money_sum(r['amount'] for r in rows), 'tds_deducted': money_sum(r['tds_amount'] for r in rows),
                'distinct_narrations': len({r['narration'] for r in rows})})
        return result

    def batch(self, limit=3, payee_id=None, offset=0):
        if type(limit) is not int or not 1 <= limit <= 5 or type(offset) is not int or offset < 0:
            raise ValueError('Batch limit must be 1–5 and group offset nonnegative')
        if payee_id is not None and payee_id not in self.data.payees:
            raise ValueError('Unknown payee identity')
        if offset and payee_id is None:
            raise ValueError('Group paging requires payee_id')
        items = []
        for identity in ([payee_id] if payee_id else self.pending()[:limit]):
            item = self.party_page(identity, offset)
            while item['nature_groups'] and len(exact_json(items + [item])) > 11500:
                item['nature_groups'].pop()
            end = offset + len(item['nature_groups'])
            item['next_group_offset'] = end if end < item['total_groups'] else None
            if not item['nature_groups'] and offset < item['total_groups']:
                if items: break
                item['requires_transaction_paging'] = True
            items.append(item)
            for group in item['nature_groups']:
                self.financial.investigation.observed.update(group['vouchers'])
        self.active = [i['payee_id'] for i in items]
        return {'items': items, 'queue': self.summary(), 'selection': 'Risk-led, with visitation fairness; omitted rows remain unprocessed.'}

    def rotate(self):
        for identity in self.active:
            self.visits[identity] += 1
            self.cooldown[identity] = self.tick + 3

    def question_priority(self, identity, fact):
        shared = 'buyer_turnover:' + self.data.client['financial_year']
        if fact != 'buyer_turnover' and shared not in self.financial.investigation.questions and not self.financial.investigation.facts:
            if len(self.financial.investigation.questions) >= 2 and any(e['goods_eligibility_impact'] > 0 for e in self.entries.values()):
                raise ValueError('One reviewer slot is reserved for shared buyer-turnover eligibility. Use queue.shared_fact_task; material scoped questions may use the other two slots.')
        if identity in self.entries:
            higher = [i for i in self.pending() if self.entries[i]['screening_upper_estimate'] > self.entries[identity]['screening_upper_estimate']]
            if higher and not self.entries[identity]['screening_upper_estimate']:
                raise ValueError('Prioritize shared buyer-turnover or other material queued facts before this low-impact screening case; it remains unprocessed.')
