#!/usr/bin/env python3
"""TDS CLI: early valid outputs, deterministic financial tools, bounded LLM loop."""
from __future__ import annotations

import argparse
import signal
from pathlib import Path

from tds_agent.output import Output


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def positive_float(value):
    import math
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="out")
    parser.add_argument("--max-calls", type=positive_int, default=48)
    parser.add_argument("--max-seconds", type=positive_int, default=600)
    parser.add_argument("--max-tools", type=positive_int, default=400)
    parser.add_argument("--max-context-chars", type=positive_int, default=100000)
    parser.add_argument("--max-output-tokens", type=positive_int, default=4000)
    parser.add_argument("--max-cost-usd", type=positive_float, default=0.25,
                        help="Stop after this reported cost; cannot cap unreported provider costs")
    parser.add_argument("--require-cost-metadata", action="store_true")
    parser.add_argument("--retrieval-mode", choices=("agent", "proactive"), default="agent")
    parser.add_argument('--investigation-mode', choices=('legacy', 'scheduled'), default='scheduled')
    args = parser.parse_args(argv)
    # The only operations before output initialization use the standard library.
    output = Output(args.out)
    agent = None
    reason, code = "initialization_failure", 1
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        output.attach_schema(Path(__file__).resolve().parent / "findings.schema.json")
        from toolkit import LLM, Toolkit
        from tds_agent.agent import Agent, Limits
        from tds_agent.tools import ToolDispatcher, FinancialTools
        from tds_agent.data import Engagement

        data = Path(args.data).resolve()
        if not data.is_dir():
            raise ValueError("--data must be an existing directory")
        # Only public metadata; never inspect private reviewer backing files.
        output.trace("configuration", data=str(data), out=str(Path(args.out).resolve()))
        kit = Toolkit(data, log_dir=output.directory)  # One instance; no quota reset.
        engagement = Engagement(data, kit, output)
        financial = FinancialTools(engagement, output, investigate=True, scheduled=args.investigation_mode == 'scheduled')
        llm = LLM(timeout=min(25, args.max_seconds))
        dispatcher = ToolDispatcher(kit, output, financial=financial)
        agent = Agent(llm, dispatcher, output, Limits(max_calls=args.max_calls, max_seconds=args.max_seconds,
                      max_tools=args.max_tools, max_context_chars=args.max_context_chars,
                      max_tokens=args.max_output_tokens, max_cost_usd=args.max_cost_usd,
                      require_cost_metadata=args.require_cost_metadata), retrieval_mode=args.retrieval_mode)
        reason = agent.run()
        # A normally ended investigation can be incomplete; only deterministic
        # coverage controls the persisted financial status, never the exit code.
        code = 0 if reason in ("phase1_finished", "financial_review_complete", "financial_review_incomplete") else 1
    except KeyboardInterrupt as exc:
        reason, code = "interrupted", 130
        output.trace("interruption", error=output.error(exc))
    except Exception as exc:
        reason = "run_failure" if agent else "initialization_failure"
        output.trace("run_error", error=output.error(exc))
    finally:
        try:
            output.finalize(reason, agent.metrics() if agent else None,
                            "Run stopped before completing the investigation." if code else None)
        finally:
            signal.signal(signal.SIGTERM, previous)
    status = "complete" if output.state["run"]["coverage_complete"] else "incomplete"
    print(f"{reason}; financial review {status}. Outputs: {output.directory}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
