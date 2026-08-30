"""CLI: Evaluate a single game log through the QUACK evaluation pipeline.

Usage:
    python scripts/evaluate_game.py game_logs/homogeneous/gpt5.5/20260308_143022_seed42/game.jsonl
    python scripts/evaluate_game.py game_logs/game_XXXXX.jsonl   # legacy flat logs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quack.evaluation.evaluator import GameEvaluator
from quack.evaluation.extraction_keys import resolve_extraction_api_key
from quack.evaluation.report import format_game_report, save_json_report
from quack.evaluation.tier3_statement_verification import extraction_output_suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single QUACK game log",
    )
    parser.add_argument(
        "log_path",
        help="Path to the game log JSONL file",
    )
    parser.add_argument(
        "--tier3",
        action="store_true",
        help="Run Tier 3 statement verification (requires LLM API key)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for the LLM used in Tier 3 claim extraction",
    )
    parser.add_argument(
        "--base-url",
        default="https://endpoint.greatrouter.com",
        help="Base URL for the LLM API endpoint",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help=(
            "LLM model for Tier 3 claim extraction (default: gpt-5.5). "
            "Also supports Claude, e.g. --model claude-opus-4-7 (routed "
            "directly to the Anthropic API; key from $ANTHROPIC_API_KEY or "
            "claude_key.txt)."
        ),
    )
    parser.add_argument(
        "--map-config",
        default="configs/maps/simple_ship.yaml",
        help="Path to the map config YAML (default: configs/maps/simple_ship.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save results as JSON to this path (default: evaluation.json next to the log)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    # Per-claim Tier 3 audit (one JSON object per extracted claim, with the
    # raw game.jsonl events the verdict was compared against) is written by
    # default whenever --tier3 runs. Pass --no-tier3-audit to skip it.
    parser.add_argument(
        "--no-tier3-audit",
        dest="save_tier3_audit",
        action="store_false",
        default=True,
        help="Skip writing tier3_claims.jsonl (default: write it next to evaluation.json)",
    )
    parser.add_argument(
        "--save-tier3-audit",
        dest="save_tier3_audit",
        action="store_true",
        default=True,
        help="(default behavior) Write tier3_claims.jsonl next to evaluation.json",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    api_key = args.api_key
    if args.tier3:
        root = Path(__file__).resolve().parent.parent
        api_key = resolve_extraction_api_key(args.model, args.api_key, root)

    evaluator = GameEvaluator(map_config_path=args.map_config)

    try:
        result = evaluator.evaluate(
            log_path=args.log_path,
            run_tier3=args.tier3,
            llm_api_key=api_key,
            llm_model=args.model,
            llm_base_url=args.base_url,
            save_tier3_audit=args.save_tier3_audit,
        )
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        sys.exit(1)

    report = format_game_report(result)
    print(report)

    # Determine output path: explicit flag > auto-detect alongside game.jsonl.
    # For a non-default Tier 3 extraction model, the auto-detected filename
    # gets a model-slug suffix (e.g. evaluation_claude-opus-4-7.json) so it
    # doesn't overwrite the default gpt-5.5 evaluation.json. An explicit
    # --output is always respected as-is.
    output_path = args.output
    if output_path is None:
        log_p = Path(args.log_path)
        if log_p.name == "game.jsonl":
            suffix = extraction_output_suffix(args.model) if args.tier3 else ""
            output_path = str(log_p.parent / f"evaluation{suffix}.json")
        else:
            output_path = None

    if output_path:
        save_json_report(result, output_path)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
