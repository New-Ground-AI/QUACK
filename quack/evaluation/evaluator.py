"""Orchestrator for running all evaluation tiers on game logs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quack.evaluation.game_reconstructor import GameReconstructor
from quack.evaluation.log_parser import get_game_config, parse_log
from quack.evaluation.tier1_game_metrics import Tier1Metrics, compute_tier1_metrics
from quack.evaluation.tier2_behavioral import Tier2Metrics, compute_tier2_metrics
from quack.evaluation.tier3_statement_verification import (
    StatementVerificationPipeline,
    Tier3Metrics,
    extraction_output_suffix,
)
from quack.map.game_map import GameMap

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single game."""

    game_id: str = ""
    log_path: str = ""
    tier1: Tier1Metrics | None = None
    tier2: Tier2Metrics | None = None
    tier3: Tier3Metrics | None = None
    tier3_audit_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result to a JSON-compatible dict."""
        result: dict[str, Any] = {
            "game_id": self.game_id,
            "log_path": self.log_path,
        }
        if self.tier1:
            result["tier1"] = self.tier1.to_dict()
        if self.tier2:
            result["tier2"] = self.tier2.to_dict()
        if self.tier3:
            tier3_dict = self.tier3.to_dict()
            if self.tier3_audit_path:
                tier3_dict["tier3_audit_path"] = self.tier3_audit_path
            result["tier3"] = tier3_dict
        return result


@dataclass
class BatchResult:
    """Aggregated evaluation results across multiple games."""

    num_games: int = 0
    results: list[EvaluationResult] = field(default_factory=list)
    aggregated: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the batch result."""
        return {
            "num_games": self.num_games,
            "aggregated": self.aggregated,
            "per_game": [r.to_dict() for r in self.results],
        }


class GameEvaluator:
    """Runs all three tiers of evaluation on a single game log."""

    def __init__(self, map_config_path: str = "configs/maps/simple_ship.yaml") -> None:
        self.map_config_path = map_config_path
        self._game_map: GameMap | None = None

    @property
    def game_map(self) -> GameMap:
        """Lazily load the game map from config."""
        if self._game_map is None:
            with open(self.map_config_path) as f:
                map_config = yaml.safe_load(f)
            self._game_map = GameMap.from_config(map_config)
        return self._game_map

    def evaluate(
        self,
        log_path: str,
        run_tier3: bool = False,
        llm_api_key: str = "",
        llm_model: str = "gpt-5.5",
        llm_base_url: str = "",
        save_tier3_audit: bool = True,
    ) -> EvaluationResult:
        """Full evaluation of a single game log.

        Args:
            log_path: Path to the JSONL game log.
            run_tier3: Whether to run the LLM-based statement verification.
            llm_api_key: API key for the LLM (required if run_tier3=True).
            llm_model: Model identifier for claim extraction.
            llm_base_url: Base URL for the LLM API.
            save_tier3_audit: When ``run_tier3`` is True, also write
                ``tier3_claims.jsonl`` (one JSON object per extracted claim:
                utterance, parsed structured claim, derived evidence, verdict,
                and the raw ``game.jsonl`` events the verdict was compared
                against). Defaults to True because the audit file is cheap
                to produce once the LLM calls have already happened. Pass
                False to opt out.

        Returns:
            EvaluationResult with all requested tiers populated.
        """
        game_id = Path(log_path).stem
        logger.info("Evaluating game: %s", game_id)

        events = parse_log(log_path)

        # Resolve map config from the log if possible
        game_config = get_game_config(events)
        map_path = game_config.get("map", self.map_config_path)
        if Path(map_path).exists():
            with open(map_path) as f:
                map_config = yaml.safe_load(f)
            game_map = GameMap.from_config(map_config)
        else:
            game_map = self.game_map

        # Tier 1
        logger.info("Computing Tier 1 metrics...")
        tier1 = compute_tier1_metrics(events)

        # Reconstruct timeline
        logger.info("Reconstructing game timeline...")
        timeline = GameReconstructor(events, game_map).reconstruct()

        # Tier 2
        logger.info("Computing Tier 2 metrics...")
        tier2 = compute_tier2_metrics(events, timeline, game_map)

        # Tier 3
        tier3 = None
        tier3_audit_path = None
        if run_tier3:
            if not llm_api_key:
                logger.warning("Tier 3 requested but no LLM API key provided; skipping")
            else:
                logger.info("Running Tier 3 statement verification...")
                # Bug C: persist the extraction cache next to the game
                # log so two runs over the same game.jsonl produce
                # byte-identical extraction (and therefore byte-identical
                # tier3_claims.jsonl + evaluation.json).
                cache_path = Path(log_path).parent / "tier3_extraction_cache.jsonl"
                pipeline = StatementVerificationPipeline(
                    events=events,
                    timeline=timeline,
                    game_map=game_map,
                    api_key=llm_api_key,
                    model=llm_model,
                    base_url=llm_base_url,
                    cache_path=cache_path,
                )
                tier3 = pipeline.run()

                # Write claim-level audit if requested. The pipeline already
                # cross-checks (Bug E) that the audit list and the metrics
                # agree before we get here, so the JSONL we write IS the
                # source the metrics were computed from.
                if save_tier3_audit and pipeline.claim_audits:
                    audit_dir = Path(log_path).parent
                    # Non-default extraction models get a model-slug suffix so
                    # e.g. a Claude run doesn't overwrite the gpt-5.5 audit.
                    suffix = extraction_output_suffix(llm_model)
                    audit_path = audit_dir / f"tier3_claims{suffix}.jsonl"
                    with open(audit_path, "w") as f:
                        for entry in pipeline.claim_audits:
                            f.write(json.dumps(entry, default=str) + "\n")
                    tier3_audit_path = str(audit_path)
                    logger.info("Tier 3 audit written to %s", tier3_audit_path)

        return EvaluationResult(
            game_id=game_id,
            log_path=str(log_path),
            tier1=tier1,
            tier2=tier2,
            tier3=tier3,
            tier3_audit_path=tier3_audit_path,
        )


class BatchEvaluator:
    """Evaluate all game logs in a directory and aggregate results."""

    def __init__(self, map_config_path: str = "configs/maps/simple_ship.yaml") -> None:
        self.map_config_path = map_config_path

    def evaluate_batch(
        self,
        log_dir: str,
        run_tier3: bool = False,
        llm_api_key: str = "",
        llm_model: str = "gpt-5.5",
        llm_base_url: str = "",
        save_tier3_audit: bool = True,
    ) -> BatchResult:
        """Evaluate all .jsonl logs in a directory.

        ``save_tier3_audit`` is forwarded to each ``GameEvaluator.evaluate``
        call so every game also produces a ``tier3_claims.jsonl`` audit file
        alongside its ``evaluation.json``.
        """
        log_files = sorted(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            logger.warning("No .jsonl files found in %s", log_dir)
            return BatchResult()

        evaluator = GameEvaluator(self.map_config_path)
        results: list[EvaluationResult] = []

        for i, log_file in enumerate(log_files, 1):
            logger.info("Processing game %d/%d: %s", i, len(log_files), log_file.name)
            try:
                result = evaluator.evaluate(
                    str(log_file),
                    run_tier3=run_tier3,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    save_tier3_audit=save_tier3_audit,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to evaluate %s: %s", log_file, e)

        aggregated = self._aggregate(results)
        return BatchResult(
            num_games=len(results),
            results=results,
            aggregated=aggregated,
        )

    def _aggregate(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Compute means and standard deviations across all games."""
        if not results:
            return {}


        agg: dict[str, Any] = {}

        # Tier 1 aggregation
        t1_metrics = [r.tier1 for r in results if r.tier1]
        if t1_metrics:
            agg["tier1"] = _aggregate_numeric_fields(
                [m.to_dict() for m in t1_metrics],
                [
                    "game_duration_ticks", "tasks_completed", "tasks_total",
                    "task_completion_rate", "total_kills", "total_meetings",
                    "total_ejections", "correct_ejections", "wrong_ejections",
                    "ejection_accuracy",
                ],
            )
            # Win rate
            winners = [m.winner for m in t1_metrics]
            agg["tier1"]["goose_win_rate"] = winners.count("goose") / len(winners)
            agg["tier1"]["duck_win_rate"] = winners.count("duck") / len(winners)

        # Tier 2 aggregation
        t2_metrics = [r.tier2 for r in results if r.tier2]
        if t2_metrics:
            agg["tier2"] = _aggregate_numeric_fields(
                [m.to_dict() for m in t2_metrics],
                [
                    "goose_voting_accuracy", "goose_skip_rate", "task_efficiency",
                    "avg_rooms_visited_goose", "avg_rooms_visited_duck",
                    "avg_kills_per_game", "avg_post_kill_displacement",
                    "self_report_rate", "cooldown_utilization",
                ],
            )

        # Tier 3 aggregation
        t3_metrics = [r.tier3 for r in results if r.tier3]
        if t3_metrics:
            agg["tier3"] = _aggregate_numeric_fields(
                [m.to_dict() for m in t3_metrics],
                [
                    "goose_truthfulness", "duck_truthfulness",
                    "spatial_hallucination_rate", "deception_rate",
                    "deception_sophistication", "accusation_accuracy",
                    "unsupported_accusation_rate", "lie_detection_rate",
                    "directly_observed_accusation_rate",
                    "testimony_derived_accusation_rate",
                    "strictly_unsupported_accusation_rate",
                ],
            )

        return agg


def _aggregate_numeric_fields(
    dicts: list[dict[str, Any]],
    fields: list[str],
) -> dict[str, Any]:
    """Compute mean and std for specified numeric fields across dicts."""
    import statistics

    result: dict[str, Any] = {}
    for field_name in fields:
        values = [
            d[field_name] for d in dicts
            if field_name in d and d[field_name] is not None
            and isinstance(d[field_name], (int, float))
        ]
        if values:
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            result[field_name] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(values)}
        else:
            result[field_name] = {"mean": None, "std": None, "n": 0}
    return result
