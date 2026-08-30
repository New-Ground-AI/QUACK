"""Generate human-readable and JSON reports from evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from quack.evaluation.evaluator import BatchResult, EvaluationResult


def format_game_report(result: EvaluationResult) -> str:
    """Generate a human-readable summary for a single game evaluation."""
    lines: list[str] = []
    lines.append(f"=== QUACK Evaluation: {result.game_id} ===")
    lines.append("")

    t1 = result.tier1
    if t1:
        lines.append("TIER 1 — Game-Level Metrics")
        lines.append(f"  Winner: {t1.winner.capitalize()} ({t1.win_reason})")
        lines.append(f"  Duration: {t1.game_duration_ticks} ticks")
        pct = t1.task_completion_rate * 100
        lines.append(f"  Tasks completed: {t1.tasks_completed}/{t1.tasks_total} ({pct:.1f}%)")
        kill_info = f"{t1.total_kills}"
        if t1.first_kill_tick is not None:
            kill_info += f" (first kill at tick {t1.first_kill_tick})"
        lines.append(f"  Kills: {kill_info}")
        lines.append(
            f"  Meetings: {t1.total_meetings} "
            f"({t1.body_report_meetings} body reports, "
            f"{t1.emergency_meetings} emergency)"
        )
        lines.append(
            f"  Ejections: {t1.correct_ejections} correct, "
            f"{t1.wrong_ejections} wrong, "
            f"{t1.no_ejection_votes} skipped"
        )
        lines.append(
            f"  Final alive: {t1.final_alive_count} "
            f"({t1.final_alive_geese} geese, {t1.final_alive_ducks} ducks)"
        )
        lines.append("")

    t2 = result.tier2
    if t2:
        lines.append("TIER 2 — Behavioral Metrics")
        lines.append(f"  Goose voting accuracy: {t2.goose_voting_accuracy * 100:.1f}%")
        lines.append(f"  Goose skip rate: {t2.goose_skip_rate * 100:.1f}%")
        if t2.avg_report_latency is not None:
            lines.append(f"  Avg report latency: {t2.avg_report_latency:.1f} ticks")
        else:
            lines.append("  Avg report latency: N/A")
        lines.append(f"  Task efficiency: {t2.task_efficiency * 100:.1f}%")
        lines.append(f"  Avg rooms visited (goose): {t2.avg_rooms_visited_goose:.1f}")
        lines.append(f"  Avg rooms visited (duck): {t2.avg_rooms_visited_duck:.1f}")
        lines.append(f"  Avg kills per game: {t2.avg_kills_per_game:.1f}")
        if t2.post_kill_displacement:
            lines.append(
                f"  Post-kill displacement: avg {t2.avg_post_kill_displacement:.1f} rooms"
            )
        else:
            lines.append("  Post-kill displacement: N/A (no kills)")
        lines.append(
            f"  Self-report rate: {t2.self_report_count} "
            f"({t2.self_report_rate * 100:.1f}%)"
        )
        lines.append(f"  Cooldown utilization: {t2.cooldown_utilization * 100:.1f}%")
        lines.append("")

    t3 = result.tier3
    if t3:
        lines.append("TIER 3 — Statement Verification")
        lines.append(f"  Claims extracted: {t3.total_claims} ({t3.verifiable_claims} verifiable)")
        lines.append(
            f"  Goose truthfulness: {t3.goose_truthfulness * 100:.1f}% "
            f"(spatial hallucination: {t3.spatial_hallucination_rate * 100:.1f}%)"
        )
        lines.append(
            f"  Duck truthfulness: {t3.duck_truthfulness * 100:.1f}% "
            f"(deception rate: {t3.deception_rate * 100:.1f}%)"
        )
        lines.append(
            f"  Deception sophistication: {t3.deception_sophistication * 100:.1f}% "
            f"(near-miss alibis)"
        )
        lines.append(f"  Lie detection rate: {t3.lie_detection_rate * 100:.1f}%")
        lines.append(f"  Accusation accuracy: {t3.accusation_accuracy * 100:.1f}%")
        lines.append(
            "  Accusation basis: "
            f"directly-observed {t3.directly_observed_accusations}, "
            f"testimony-derived {t3.testimony_derived_accusations}, "
            f"unsupported {t3.unsupported_accusations} "
            f"(unsupported rate: {t3.strictly_unsupported_accusation_rate * 100:.1f}%)"
        )

        if t3.claim_type_distribution:
            dist_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(t3.claim_type_distribution.items())
            )
            lines.append(f"  Claim types: {dist_str}")
        lines.append("")

    return "\n".join(lines)


def format_batch_report(batch: BatchResult) -> str:
    """Generate a human-readable summary for batch evaluation."""
    lines: list[str] = []
    lines.append(f"=== QUACK Batch Evaluation ({batch.num_games} games) ===")
    lines.append("")

    agg = batch.aggregated
    if not agg:
        lines.append("No results to aggregate.")
        return "\n".join(lines)

    if "tier1" in agg:
        lines.append("TIER 1 — Aggregated Game-Level Metrics")
        t1 = agg["tier1"]
        if "goose_win_rate" in t1:
            lines.append(f"  Goose win rate: {t1['goose_win_rate'] * 100:.1f}%")
            lines.append(f"  Duck win rate: {t1['duck_win_rate'] * 100:.1f}%")
        for key in [
            "game_duration_ticks", "task_completion_rate", "total_kills",
            "ejection_accuracy",
        ]:
            if key in t1 and t1[key]["mean"] is not None:
                label = key.replace("_", " ").title()
                lines.append(
                    f"  {label}: {t1[key]['mean']:.2f} ± {t1[key]['std']:.2f}"
                )
        lines.append("")

    if "tier2" in agg:
        lines.append("TIER 2 — Aggregated Behavioral Metrics")
        t2 = agg["tier2"]
        for key in [
            "goose_voting_accuracy", "goose_skip_rate", "task_efficiency",
            "avg_rooms_visited_goose", "avg_rooms_visited_duck",
            "avg_kills_per_game", "cooldown_utilization",
        ]:
            if key in t2 and t2[key]["mean"] is not None:
                label = key.replace("_", " ").title()
                lines.append(
                    f"  {label}: {t2[key]['mean']:.2f} ± {t2[key]['std']:.2f}"
                )
        lines.append("")

    if "tier3" in agg:
        lines.append("TIER 3 — Aggregated Statement Verification")
        t3 = agg["tier3"]
        for key in [
            "goose_truthfulness", "duck_truthfulness",
            "spatial_hallucination_rate", "deception_rate",
            "accusation_accuracy", "lie_detection_rate",
        ]:
            if key in t3 and t3[key]["mean"] is not None:
                label = key.replace("_", " ").title()
                lines.append(
                    f"  {label}: {t3[key]['mean']:.2f} ± {t3[key]['std']:.2f}"
                )
        lines.append("")

    return "\n".join(lines)


def save_json_report(result: EvaluationResult | BatchResult, output_path: str) -> None:
    """Save evaluation results as a JSON file."""
    data = result.to_dict()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
