"""Strict dispatch around the unmodified Toolkit tools."""
from __future__ import annotations

import json
import copy
import hashlib
from pathlib import Path

from .output import INCOMPLETE, exact_json, strict_json


def envelope(success, data, error_code=None):
    result = {"success": success, "data": data}
    if error_code:
        result["error_code"] = error_code
    return exact_json(result)


def source_text(value):
    """Preserve actual reviewer response strings, including labeled fact lines."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(source_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(source_text(v) for v in value)
    return ""


class ToolDispatcher:
    def __init__(self, kit, output, max_result_chars=18000, financial=None):
        from jsonschema import Draft202012Validator

        self.kit, self.output = kit, output
        self.max_result_chars = max_result_chars
        self.financial = financial
        self.specs = list(kit.specs) + (financial.specs if financial else [])
        self.functions = dict(kit.functions)
        if financial:
            self.functions.update(financial.functions)
        self.investigation = financial.investigation if financial else None
        if self.investigation:
            self.functions["read_document"] = lambda **args: (envelope(True, self.investigation.document_page(**args))
                if args["doc_id"] in self.financial.data.document_index else kit.functions["read_document"](**args))
            self.functions["search_documents"] = lambda **args: envelope(True, self.investigation.search(**args))
            self.specs = copy.deepcopy(self.specs)
            reviewer_spec = next(s["function"] for s in self.specs if s["function"]["name"] == "ask_reviewer")
            reviewer_spec["parameters"]["properties"].update({"payee_id": STRING,
                "fact": {"enum": ["buyer_turnover", "transporter_eligibility", "payment_nature", "identity"]}})
            reviewer_spec["parameters"]["required"] = ["question", "why", "fact"]
        self.validators = {s["function"]["name"]: Draft202012Validator(s["function"]["parameters"])
                           for s in self.specs}

    def execute(self, call_id, name, arguments, blocked=None, result_char_budget=None):
        self.output.trace("tool_start", tool_call_id=call_id, name=name, arguments=arguments)
        args = None
        try:
            if blocked:
                result = envelope(False, blocked, "RUN_LIMIT")
            elif name not in self.validators or name not in self.functions:
                result = envelope(False, "Unknown tool name", "UNKNOWN_TOOL")
            else:
                args = strict_json(arguments)
                self.validators[name].validate(args)
                self._bounds(name, args)
                delivery_before = (set(self.investigation.observed), copy.deepcopy(self.investigation.delivered)) if self.investigation else None
                if self.investigation and name == "ask_reviewer":
                    key = self.investigation.question_key(args)
                    if key in self.investigation.questions:
                        raw = self.investigation.questions[key]
                        self.output.trace("reviewer_cache_hit", tool_call_id=call_id, fact_key=key)
                        self.output.trace("tool_result", tool_call_id=call_id, name=name, arguments=args, result=strict_json(raw))
                        return self.output.clean(raw)
                    if len(self.investigation.questions) >= 3:
                        result = envelope(False, "Reviewer question allowance exhausted; preserve the specific unresolved financial fact.", "REVIEWER_BUDGET")
                        self.output.trace("tool_result", tool_call_id=call_id, name=name, arguments=args, result=strict_json(result))
                        return result
                    # Reserve once before executing; an exception never creates a
                    # free retry or resets the Toolkit's own question allowance.
                    self.investigation.questions[key] = envelope(False, "Reviewer attempt failed; no answer available", "REVIEWER_UNAVAILABLE")
                    value = self.kit.functions[name](question=self.output.clean(args["question"]), why=self.output.clean(args["why"]))
                else:
                    value = self.functions[name](**args)
                raw = envelope(True, value) if self.financial and name in self.financial.functions else value
                parsed = strict_json(raw)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("success"), bool) or "data" not in parsed:
                    raise ValueError("Tool returned an invalid result envelope")
                if self.financial and name == "ask_reviewer" and parsed["success"]:
                    reference = "reviewer:" + call_id
                    answer = parsed["data"].get("answer", "") if self.investigation and isinstance(parsed["data"], dict) else parsed["data"]
                    self.financial.data.review_evidence[reference] = source_text(answer)
                    if self.investigation:
                        self.investigation.review_records[reference] = {**args, "response": parsed["data"]}
                    parsed["data"] = {"response": parsed["data"], "evidence_id": reference}
                    raw = exact_json(parsed)
                    if self.investigation:
                        self.investigation.questions[key] = raw
                self.output.trace("tool_result", tool_call_id=call_id, name=name, arguments=args, result=parsed)
                delivery_limit = min(self.max_result_chars, result_char_budget) if result_char_budget is not None else self.max_result_chars
                if len(raw) > delivery_limit:
                    if delivery_before:
                        self.investigation.observed, self.investigation.delivered = delivery_before
                    # Full result stays in trace. No silently truncated financial data.
                    result = envelope(False, f"Result exceeds this batch's {delivery_limit}-character delivery budget. "
                                      "Request fewer tools per reply, smaller summary/transaction limits, or smaller document pages. "
                                      "Full result is retained in trace; hidden evidence is not marked delivered.", "RESULT_TOO_LARGE")
                    self.output.trace("tool_delivery", tool_call_id=call_id, result=strict_json(result))
                    return self.output.clean(result)
                else:
                    return self.output.clean(raw)
        except KeyboardInterrupt as exc:
            self.output.trace("tool_result", tool_call_id=call_id, name=name, arguments=args,
                              result={"success": False, "data": self.output.error(exc), "error_code": "INTERRUPTED"})
            raise
        except (ValueError, TypeError) as exc:
            result = envelope(False, self.output.error(exc), "INVALID_ARGUMENTS")
        except Exception as exc:
            # ValidationError is deliberately kept out of tracebacks and returned to
            # the model in the same envelope as ordinary execution errors.
            code = "INVALID_ARGUMENTS" if type(exc).__name__ == "ValidationError" else "TOOL_ERROR"
            result = envelope(False, self.output.error(exc), code)
        self.output.trace("tool_result", tool_call_id=call_id, name=name, arguments=args,
                          result=strict_json(result))
        return self.output.clean(result)

    def _bounds(self, name, args):
        for key, upper in (("offset", 10_000_000), ("limit", 100 if name in ("get_party_summary", "get_party_transactions") else 6000), ("top_k", 10)):
            if key in args and not (0 if key == "offset" else 1) <= args[key] <= upper:
                raise ValueError(f"{key} outside allowed range")
        for value in args.values():
            if isinstance(value, str) and len(value) > 4000:
                raise ValueError("Argument string exceeds 4000 characters")
        # Toolkit exposes IDs rather than arbitrary paths. Reject traversal even
        # though its current implementation also uses an allowlisted document map.
        if name == "read_document":
            doc_id = args["doc_id"]
            if Path(doc_id).is_absolute() or ".." in Path(doc_id).parts or "\\" in doc_id:
                raise ValueError("Document must be an engagement document ID")


def function_spec(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}}}


STRING = {"type": "string", "minLength": 1, "maxLength": 4000}
STRINGS = {"type": "array", "items": STRING, "minItems": 1, "maxItems": 10000, "uniqueItems": True}
PAGINATION = {"offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}


class FinancialTools:
    """Payee-wide classification plans prevent partial threshold calculations.

    LLM supplies classifications and source IDs, never amounts/rates. Every plan
    must partition all of a payee's vouchers exactly once before calculation.
    Only record_assessment changes reviewed coverage or persisted findings.
    """
    def __init__(self, data, output, investigate=False, scheduled=False):
        from .calculations import Calculator, CATEGORIES

        self.data, self.output, self.calculator = data, output, Calculator(data)
        from .investigation import Investigation
        self.investigation = Investigation(data, output) if investigate else None
        self.calculations, self.plans = {}, {}
        self.scheduler = None
        self.external_limitations = []
        classification = {"type": "object", "properties": {
            "section": {"enum": list(CATEGORIES)}, "category": STRING,
            "vouchers": STRINGS, "ledgers": STRINGS, "evidence": STRINGS,
            "needs": STRING, "certificate_ids": STRINGS, "transporter_id": STRING, "turnover_id": STRING},
            "required": ["section", "category", "evidence"], "additionalProperties": False,
            "oneOf": [{"required": ["vouchers"], "not": {"required": ["ledgers"]}},
                      {"required": ["ledgers"], "not": {"required": ["vouchers"]}}]}
        classification["properties"]["citations"] = {"type": "array", "minItems": 1, "maxItems": 30,
            "items": {"type": "object", "properties": {"source_id": STRING, "quote": {"type": "string", "minLength": 3, "maxLength": 1500}},
                      "required": ["source_id", "quote"], "additionalProperties": False}}
        if self.investigation:
            classification["required"].append("citations")
        self.specs = [
            function_spec("get_party_summary", "Paginated full-ledger PAN/payee totals and actual coverage. Ledger names are views, not corporate groups.",
                          {"payee_id": STRING, **PAGINATION}),
            function_spec("get_party_transactions", "Exact stored rows for a payee. Follow next_offset until null; no samples.",
                          {"payee_id": STRING, **PAGINATION}, ["payee_id"]),
            function_spec("calculate_tds", "Submit a complete payee classification plan partitioning all vouchers once (by vouchers OR exact ledger names). "
                          "Python aggregates by section/category, reads YAML rates and source document exception facts, and returns calculation IDs. "
                          "Use unknown + a specific needs statement for unresolved financial facts; excluded only for evidence-backed out-of-scope nature. "
                          "Evidence IDs must be stored voucher numbers, public document IDs or reviewer evidence IDs. No monetary arguments accepted.",
                          {"payee_id": STRING, "classifications": {"type": "array", "items": classification, "minItems": 1, "maxItems": 1000}},
                          ["payee_id", "classifications"]),
            function_spec("record_assessment", "Record an existing deterministic calculation, with a source-supported explanation. "
                          "Numbers/status cannot be overridden. Compliant/excluded cases stay in coverage; exceptions checkpoint into findings.",
                          {"calculation_id": STRING, "reason": {**STRING, "minLength": 10}}, ["calculation_id", "reason"])]
        self.functions = {"get_party_summary": data.get_party_summary, "get_party_transactions": data.get_party_transactions,
                          "calculate_tds": self.calculate_tds, "record_assessment": self.record_assessment}
        if self.investigation:
            self.functions.update({"get_party_summary": self.investigation.summary,
                                   "get_party_transactions": self.investigation.transactions,
                                   "record_evidence": self.record_evidence})
            self.specs.append(function_spec("record_evidence", "Validate exact quoted turnover or year-wide transporter declaration facts from source evidence. "
                                           "Python parses values, checks scope, and returns a calculator-ready evidence ID. No numeric arguments.",
                                           {"kind": {"enum": ["buyer_turnover", "transporter"]}, "source_id": STRING, "payee_id": STRING,
                                            "quote": {"type": "string", "minLength": 5, "maxLength": 2000}}, ["kind", "source_id", "quote"]))
        if scheduled:
            if not self.investigation:
                raise ValueError('Scheduled investigation requires evidence validation')
            from .scheduler import Scheduler
            from jsonschema import Draft202012Validator
            self.scheduler = Scheduler(self)
            self.investigation.scheduler = self.scheduler
            entry = {'type': 'object', 'properties': {'payee_id': STRING,
                     'classifications': {'type': 'array', 'items': classification, 'minItems': 1, 'maxItems': 30},
                     'reason': {**STRING, 'minLength': 10}, 'defer_reason': STRING},
                     'required': ['payee_id'], 'additionalProperties': False,
                     'oneOf': [{'required': ['classifications', 'reason'], 'not': {'required': ['defer_reason']}},
                               {'required': ['defer_reason'], 'not': {'anyOf': [{'required': ['classifications']}, {'required': ['reason']}]}}]}
            self.batch_validator = Draft202012Validator(entry)
            self.functions.update(get_investigation_batch=self.scheduler.batch, assess_batch=self.assess_batch)
            self.specs.extend([
                function_spec('get_investigation_batch', 'Risk-led complete nature groups with original vouchers and representative exact quotes. '
                              'Not liability/compliance. Follow next_group_offset with payee_id+offset; huge groups require get_party_transactions.',
                              {'payee_id': STRING, 'offset': {'type': 'integer', 'minimum': 0},
                               'limit': {'type': 'integer', 'minimum': 1, 'maximum': 5}}),
                function_spec('assess_batch', 'Calculate AND checkpoint up to 5 independent complete payee plans in one call. '
                              'Each entry is validated independently; rejected entries stay unprocessed. Existing calculator/citation checks apply. '
                              'For each entry use payee_id,classifications,reason OR payee_id,defer_reason. No monetary inputs. '
                              'Classifications use the exact calculate_tds schema (shown below); split categories explicitly. '
                              'Do not classify uncertain consultancy as professional merely from its booked section.',
                              {'assessments': {'type': 'array', 'minItems': 1, 'maxItems': 5,
                                               'items': {'description': 'Individually validated entry', **entry,
                                                         # Validate each item inside the tool so a bad item
                                                         # cannot discard independent valid assessments.
                                                         'oneOf': entry['oneOf']}}}, ['assessments'])])
            # Keep the detailed entry contract in the description, but permit
            # per-item validation failures to return alongside accepted items.
            self.specs[-1]['function']['description'] += ' Entry contract: ' + exact_json(entry)
            self.specs[-1]['function']['parameters']['properties']['assessments']['items'] = {}
        output.financial = self
        output.checkpoint(self.snapshot())
        output.trace("financial_tools_ready", phase=2, coverage=data.coverage_report())

    def assess_batch(self, assessments):
        if not isinstance(assessments, list) or not 1 <= len(assessments) <= 5:
            raise ValueError('Provide 1–5 independent assessment entries')
        results, seen = [], set()
        for item in assessments:
            recorded = []
            identity = item.get('payee_id') if isinstance(item, dict) else None
            try:
                self.batch_validator.validate(item)
                if identity not in self.data.payees or identity in seen:
                    raise ValueError('Unknown or duplicate batch payee identity')
                seen.add(identity)
                if 'defer_reason' in item:
                    self.scheduler.deferred[identity] = item['defer_reason']
                    results.append({'payee_id': identity, 'success': True, 'deferred': True, 'coverage_changed': False})
                    continue
                calculated = self.calculate_tds(identity, item['classifications'])
                for calculation in calculated['calculations']:
                    calculation_id = calculation['calculation_id']
                    if calculation_id not in self.data.assessments:
                        self.record_assessment(calculation_id, item['reason'])
                    recorded.append(calculation_id)
                self.scheduler.deferred.pop(identity, None)
                results.append({'payee_id': identity, 'success': True, 'recorded': recorded,
                                'calculations': calculated['calculations']})
            except Exception as exc:
                from .calculations import CATEGORIES
                error = {'type': type(exc).__name__, 'message': str(getattr(exc, 'message', str(exc)))[:900]}
                results.append({'payee_id': identity, 'success': False, 'error_code': 'ASSESSMENT_REJECTED',
                                'error': self.output.clean(error), 'recorded_before_failure': recorded,
                                'allowed_section_categories': {k: sorted(v) for k, v in CATEGORIES.items()},
                                'recovery': 'Retry rejected payees only. Copy exact batch quote/source_voucher; unknown/unknown requires needs; excluded/excluded requires evidence. Accepted entries stay recorded.'})
        return {'results': results, 'queue': self.scheduler.summary(),
                'coverage': {k: v for k, v in self.data.coverage_report().items() if k != 'groups'}}

    def record_evidence(self, kind, source_id, quote, payee_id=None):
        if kind == "buyer_turnover":
            return self.investigation.register_turnover(source_id, quote)
        return self.investigation.register_transporter(source_id, quote, payee_id)

    def calculate_tds(self, payee_id, classifications):
        if payee_id not in self.data.payees:
            raise ValueError("Unknown payee identity")
        signature = exact_json(classifications)
        if payee_id in self.plans:
            prior_signature, ids = self.plans[payee_id]
            if prior_signature == signature:
                return {"calculations": [self.compact(self.calculations[i]) for i in ids]}
            if any(c["payee_id"] == payee_id for c in self.data.assessments.values()):
                raise ValueError("Cannot reclassify a recorded payee in this run; existing findings are preserved")
        payee_vouchers = self.data.payees[payee_id]["vouchers"]
        selections, seen = [], set()
        for item in classifications:
            item = copy.deepcopy(item)
            if "ledgers" in item:
                ledgers = item.pop("ledgers")
                available = {self.data.transactions[v]["ledger"] for v in payee_vouchers}
                if not set(ledgers) <= available:
                    raise ValueError("Unknown ledger name for this payee")
                item["vouchers"] = [v for v in payee_vouchers if self.data.transactions[v]["ledger"] in ledgers]
            selected = item["vouchers"]
            self.data.select(selected)
            if not set(selected) <= set(payee_vouchers) or seen.intersection(selected):
                raise ValueError("Classification selections overlap or cross payee boundaries")
            seen.update(selected)
            if self.investigation:
                item["_citations"] = self.investigation.validate_classification(payee_id, item, selected)
                for doc_id in item.get("certificate_ids", []) + [item[k] for k in ("transporter_id", "turnover_id") if item.get(k)]:
                    self.investigation.source(doc_id)
                for reference in (item.get("turnover_id"), item.get("transporter_id")):
                    fact = self.investigation.facts.get(reference)
                    if fact:
                        item["evidence"] = list(dict.fromkeys(item["evidence"] + [fact["source_id"]]))
            else:
                item.pop("citations", None)
            selections.append(item)
        if seen != set(payee_vouchers):
            raise ValueError("Incomplete classification plan: every payee voucher must be assigned exactly once")
        # Merge identical financial categories before threshold evaluation. Different
        # narratives or ledger labels never split one category's annual threshold.
        buckets = {}
        for item in selections:
            key = (item["section"], item["category"], item.get("needs") if item["section"] == "unknown" else None)
            if key not in buckets:
                buckets[key] = item
                continue
            target = buckets[key]
            for source_key in ("transporter_id", "turnover_id"):
                if target.get(source_key) != item.get(source_key):
                    raise ValueError("Conflicting exception sources within the same section/category")
            for array_key in ("vouchers", "evidence", "certificate_ids"):
                target[array_key] = list(dict.fromkeys(target.get(array_key, []) + item.get(array_key, [])))
            if "_citations" in item:
                target.setdefault("_citations", []).extend(item["_citations"])
        rent = [v for item in buckets.values() if item["section"] == "194I" for v in item["vouchers"]]
        section_certificates = {}
        for item in buckets.values():
            section_certificates.setdefault(item["section"], set()).update(item.get("certificate_ids", []))
        results = []
        for item in buckets.values():
            if section_certificates[item["section"]]:
                item["certificate_ids"] = sorted(section_certificates[item["section"]])
            citations = item.pop("_citations", [])
            result = self.calculator.calculate(**item, rent_vouchers=rent or None)
            if citations:
                result["classification_citations"] = citations
            result["calculation_id"] = "calc_" + hashlib.sha256(exact_json(result).encode()).hexdigest()[:20]
            results.append(result)
        # All calculations succeed or no plan is committed. Old IDs are invalidated
        # on an unrecorded reclassification; no stale monetary results are accepted.
        if payee_id in self.plans:
            for calc_id in self.plans[payee_id][1]:
                self.calculations.pop(calc_id, None)
        ids = []
        for result in results:
            ids.append(result["calculation_id"])
            self.calculations[result["calculation_id"]] = result
            self.output.trace("financial_calculation", calculation=result)
        self.plans[payee_id] = (signature, ids)
        return {"calculations": [self.compact(result) for result in results],
                "coverage_changed": False, "next": "record_assessment for every returned calculation_id"}

    @staticmethod
    def compact(result):
        keys = ("calculation_id", "payee_id", "section", "category", "status", "amount_total", "amount_liable",
                "tds_required", "tds_deducted", "shortfall", "needs", "rule_references", "evidence")
        return {**{k: result[k] for k in keys if k in result}, "voucher_count": len(result["vouchers"])}

    def snapshot(self, assessments=None, coverage=None):
        assessments = self.data.assessments if assessments is None else assessments
        coverage = self.data.coverage if coverage is None else coverage
        findings = []
        for result in assessments.values():
            if result["status"] in ("compliant", "excluded"):
                continue
            finding = {k: result[k] for k in ("party", "pan", "section", "status", "shortfall", "reason", "needs",
                                             "amount_liable", "tds_required", "tds_deducted", "vouchers") if k in result}
            finding["evidence"] = list(dict.fromkeys(result["evidence"] + result["rule_references"]))
            findings.append(finding)
        counts = {s: sum(state == s for state in coverage.values()) for s in ("unprocessed", "unresolved", "assessed", "compliant")}
        complete = bool(coverage) and not self.data.limitations and all(s in ("assessed", "compliant") for s in coverage.values())
        limitations = list(dict.fromkeys(self.data.limitations + self.external_limitations))
        if not complete:
            limitations.append(f"Incomplete financial coverage: {counts['unprocessed']} unprocessed and {counts['unresolved']} unresolved vouchers out of {len(coverage)}.")
        return {"client": self.data.client["company"], "financial_year": self.data.client["financial_year"],
                "findings": findings, "limitations": limitations,
                "run": {"run_id": self.output.run_id, "status": "complete" if complete else "incomplete", "coverage_complete": complete}}

    def record_assessment(self, calculation_id, reason):
        if calculation_id not in self.calculations:
            raise ValueError("Unknown or stale calculation ID")
        result = copy.deepcopy(self.calculations[calculation_id])
        result["reason"] = f"{result['section']}/{result['category']}: {reason}"
        if not reason.strip() or len(reason.strip()) < 10:
            raise ValueError("Provide a specific source-supported assessment explanation")
        if calculation_id in self.data.assessments:
            existing = self.data.assessments[calculation_id]
            self.output.trace('assessment_reused', calculation_id=calculation_id)
            return {'calculation_id': calculation_id, 'status': existing['status'], 'already_recorded': True,
                    'coverage': {k: v for k, v in self.data.coverage_report().items() if k != 'groups'}}
        assessments = {**self.data.assessments, calculation_id: result}
        coverage = dict(self.data.coverage)
        state = "unresolved" if result["status"] == "needs_review" else ("compliant" if result["status"] in ("compliant", "excluded") else "assessed")
        for voucher in result["vouchers"]:
            coverage[voucher] = state
        candidate = self.snapshot(assessments, coverage)
        try:
            self.output.checkpoint(candidate)
        finally:
            # Keep the in-memory financial state aligned even if interruption or
            # trace failure occurs immediately after a successful atomic write.
            if self.output.state == self.output.clean(candidate):
                self.data.assessments, self.data.coverage = assessments, coverage
        self.output.trace("assessment_recorded", calculation_id=calculation_id, assessment=result,
                          coverage=self.data.coverage_report())
        return {"calculation_id": calculation_id, "status": result["status"], "coverage":
                {k: v for k, v in self.data.coverage_report().items() if k != "groups"}}
