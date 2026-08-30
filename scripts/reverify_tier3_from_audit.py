"""Offline re-verify: re-run Tier 3 verification on the structured claims
already present in existing ``tier3_claims.jsonl`` audit files (skipping the
LLM extraction step).

Use this to demonstrate the impact of Tier 3 verifier-side changes (e.g. the
directly-observed / testimony-derived / unsupported accusation taxonomy, or
Bug A/B/D/F fixes) without re-paying for LLM extraction. The extraction step
(cache / dedup) is a separate concern and not exercised here.

Works on either a single run directory OR a whole setting directory:

- **Single run** — a directory that directly contains ``tier3_claims.jsonl``.
- **Batch** — any parent directory; every run under it (found recursively by
  its ``tier3_claims.jsonl``) is re-verified, and an aggregate accusation
  summary across all runs is printed and written to
  ``<root>/reverify_summary.json``.

Outputs written alongside each run directory:
- ``tier3_claims_reverified.jsonl`` — same structured claims, new verdicts
- ``evaluation_reverified.json``   — Tier 3 metrics computed from the new verdicts

Run::

    # one run
    python scripts/reverify_tier3_from_audit.py game_logs/homogeneous/gpt5.5/<run>/

    # every run in a setting
    python scripts/reverify_tier3_from_audit.py game_logs/homogeneous/gpt5.5/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quack.evaluation.game_reconstructor import GameReconstructor
from quack.evaluation.log_parser import (
    get_game_config, get_initial_state, get_player_role_map, parse_log,
)
from quack.evaluation.tier3_statement_verification import (
    StatementVerificationPipeline, Tier3Metrics,
    _canonical_claim_signature,
)
from quack.map.game_map import GameMap


def _load_game_map(events: list[dict], default_map_yaml: Path) -> GameMap:
    """Resolve the game map for a run, preferring the map recorded in the log
    (matching ``GameEvaluator``), falling back to ``default_map_yaml``.
    """
    map_path = get_game_config(events).get("map")
    if map_path and Path(map_path).exists():
        with open(map_path) as f:
            return GameMap.from_config(yaml.safe_load(f))
    with open(default_map_yaml) as f:
        return GameMap.from_config(yaml.safe_load(f))


def reverify_run(
    run_dir: Path,
    map_yaml: Path,
) -> tuple[Tier3Metrics, dict[str, int]] | None:
    """Re-verify a single run directory.

    Returns ``(metrics, verdict_diff)`` on success, or ``None`` if the run has
    no ``tier3_claims.jsonl`` to re-verify. Writes
    ``tier3_claims_reverified.jsonl`` + ``evaluation_reverified.json`` into
    ``run_dir``.
    """
    game_log = run_dir / "game.jsonl"
    audit_path = run_dir / "tier3_claims.jsonl"
    if not audit_path.exists():
        return None

    events = parse_log(str(game_log))
    game_map = _load_game_map(events, map_yaml)
    timeline = GameReconstructor(events, game_map).reconstruct()

    pipeline = StatementVerificationPipeline.__new__(StatementVerificationPipeline)
    pipeline.events = events
    pipeline.timeline = timeline
    pipeline.game_map = game_map
    initial = get_initial_state(events)
    pipeline.role_map = get_player_role_map(events)
    pipeline.name_to_id = {info["name"]: pid for pid, info in initial.items()}
    pipeline.id_to_name = {pid: info["name"] for pid, info in initial.items()}
    pipeline.player_names = list(pipeline.id_to_name.values())
    pipeline.duck_ids = {pid for pid, t in pipeline.role_map.items() if t == "duck"}
    pipeline.claim_audits = []

    audits_in: list[dict] = []
    with open(audit_path) as f:
        for line in f:
            if line.strip():
                audits_in.append(json.loads(line))

    # Group existing audits by meeting so we can rebuild meeting context.
    meetings_by_idx: dict[int, dict] = {}
    for a in audits_in:
        midx = a["meeting"]["meeting_idx"]
        if midx not in meetings_by_idx:
            meetings_by_idx[midx] = {
                "tick": a["meeting"]["meeting_tick"],
                "type": a["meeting"]["meeting_type"],
                "caller": a["meeting"]["caller_id"],
            }

    # Apply dedup to the input audits before re-verifying, so the output
    # mirrors what a full re-run of the pipeline (with extraction cache +
    # dedup) would produce. Duplicates from the OLD pipeline get collapsed
    # here. ``dedup_collapsed_n`` is summed per (speaker, meeting) so the
    # surviving entry carries the dropped count.
    seen: set[tuple[str, int, tuple]] = set()
    collapsed_by_key: dict[tuple[str, int], int] = {}
    deduped_in: list[dict] = []
    for a in audits_in:
        sc = a["structured_claim"]
        sig = _canonical_claim_signature({
            "type": sc["claim_type"],
            "subject": sc.get("subject", ""),
            "target": sc.get("target", ""),
            "room": sc.get("room") or "",
            "activity": sc.get("activity") or "",
            "temporal": sc.get("temporal_ref") or "",
        })
        key_full = (a["speaker"]["speaker_id"], a["meeting"]["meeting_idx"], sig)
        key_partial = (a["speaker"]["speaker_id"], a["meeting"]["meeting_idx"])
        if key_full in seen:
            collapsed_by_key[key_partial] = collapsed_by_key.get(key_partial, 0) + 1
            continue
        seen.add(key_full)
        deduped_in.append(a)

    audits_in = deduped_in

    all_verified: list[dict] = []
    new_audits: list[dict] = []
    meeting_duck_lies: dict[int, bool] = {}
    meeting_duck_caught: dict[int, bool] = {}
    verdict_diff: dict[str, int] = {}

    for a in audits_in:
        sc = a["structured_claim"]
        midx = a["meeting"]["meeting_idx"]
        meeting_tick = a["meeting"]["meeting_tick"]
        speaker_id = a["speaker"]["speaker_id"]
        speaker_name = a["speaker"]["speaker_name"]

        claim = {
            "type": sc["claim_type"],
            "subject": sc.get("subject", ""),
            "target": sc.get("target"),
            "room": sc.get("room"),
            "activity": sc.get("activity"),
            "temporal": sc.get("temporal_ref", "this round"),
            "_speaker_id": speaker_id,
            "_speaker_name": speaker_name,
            "_meeting_idx": midx,
            "_meeting_tick": meeting_tick,
        }
        if sc.get("route"):
            claim["route"] = sc["route"]

        old_verdict = a["verification"]["verdict"]
        new_result = pipeline._verify_claim(claim, meeting_tick)
        new_verdict = new_result.verdict
        verdict_diff[f"{old_verdict}->{new_verdict}"] = (
            verdict_diff.get(f"{old_verdict}->{new_verdict}", 0) + 1
        )
        claim["_verdict"] = new_verdict
        claim["_verification"] = new_result
        all_verified.append(claim)

        meeting = meetings_by_idx[midx]
        raw_utt = a["utterance"]["raw"]
        new_audit = pipeline._build_audit_entry(
            claim, meeting, midx, new_result, raw_utt,
        )
        # Preserve provenance metadata (cache_hit etc.) from the old audit
        # if present, otherwise tag as a reverify run. Stamp the dedup
        # count we computed above onto every audit entry for that
        # (speaker, meeting).
        new_audit["extraction"] = a.get("extraction") or {
            "cache_hit": True, "dedup_collapsed_n": 0,
            "prompt_version": "reverify-no-llm-call",
            "model": "<not-called>",
        }
        new_audit["extraction"]["dedup_collapsed_n"] = (
            collapsed_by_key.get((speaker_id, midx), 0)
        )
        new_audit["reverified_from"] = {
            "old_verdict": old_verdict,
            "new_verdict": new_verdict,
        }
        new_audits.append(new_audit)
        pipeline.claim_audits.append(new_audit)

    # Build meeting_duck_lies / caught from new verdicts
    for midx in meetings_by_idx:
        had_lies = any(
            c["_speaker_id"] in pipeline.duck_ids
            and c["_verdict"] in ("false", "wrong_room")
            for c in all_verified if c["_meeting_idx"] == midx
        )
        meeting_duck_lies[midx] = had_lies
        meeting_duck_caught[midx] = False

    metrics = Tier3Metrics()
    pipeline._compute_metrics(
        metrics, all_verified, meeting_duck_lies, meeting_duck_caught,
    )

    out_audit = run_dir / "tier3_claims_reverified.jsonl"
    out_eval = run_dir / "evaluation_reverified.json"
    with open(out_audit, "w") as f:
        for entry in new_audits:
            f.write(json.dumps(entry, default=str) + "\n")
    with open(out_eval, "w") as f:
        json.dump({"tier3": metrics.to_dict()}, f, indent=2, default=str)

    return metrics, verdict_diff


def _print_accusation_summary(metrics: Tier3Metrics, indent: str = "  ") -> None:
    total = metrics.total_accusations
    print(f"{indent}total accusations:       {total:>4}")
    print(f"{indent}directly-observed:       {metrics.directly_observed_accusations:>4} "
          f"({metrics.directly_observed_accusation_rate * 100:.1f}%)")
    print(f"{indent}testimony-derived:       {metrics.testimony_derived_accusations:>4} "
          f"({metrics.testimony_derived_accusation_rate * 100:.1f}%)")
    print(f"{indent}unsupported:             {metrics.unsupported_accusations:>4} "
          f"({metrics.strictly_unsupported_accusation_rate * 100:.1f}%)")
    print(f"{indent}accusation accuracy:     {metrics.accusation_accuracy * 100:.1f}%")


def _discover_run_dirs(root: Path) -> list[Path]:
    """Find every run directory under ``root`` that has a tier3_claims.jsonl.

    ``rglob`` includes ``root`` itself, so a single run directory passed
    directly is handled the same way (yields exactly one match).
    """
    return sorted({p.parent for p in root.rglob("tier3_claims.jsonl")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=str,
        help="A run directory (with game.jsonl + tier3_claims.jsonl) OR a "
             "parent/setting directory containing many such runs",
    )
    parser.add_argument("--map-config", type=str,
                        default="configs/maps/simple_ship.yaml",
                        help="Fallback map config YAML (used when a run's log "
                             "doesn't record a resolvable map path)")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"ERROR: path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    map_yaml = Path(args.map_config)
    run_dirs = _discover_run_dirs(root)
    if not run_dirs:
        print(f"ERROR: no tier3_claims.jsonl found under {root}", file=sys.stderr)
        print("       (run --tier3 evaluation first to produce the audit files)",
              file=sys.stderr)
        sys.exit(1)

    batch = len(run_dirs) > 1
    print(f"Found {len(run_dirs)} run(s) with tier3_claims.jsonl under {root}")
    print("=" * 60)

    # Aggregators across all runs.
    agg_verdict_diff: dict[str, int] = {}
    agg = Tier3Metrics()
    per_run_summary: list[dict[str, Any]] = []
    processed = 0
    failed = 0

    for run_dir in run_dirs:
        rel = run_dir.relative_to(root) if run_dir != root else Path(".")
        try:
            result = reverify_run(run_dir, map_yaml)
        except Exception as e:  # keep going across a batch
            print(f"[FAIL] {rel}: {e}")
            failed += 1
            continue
        if result is None:
            continue
        metrics, verdict_diff = result
        processed += 1

        for k, v in verdict_diff.items():
            agg_verdict_diff[k] = agg_verdict_diff.get(k, 0) + v

        # Sum the accusation counters we aggregate on.
        agg.total_accusations += metrics.total_accusations
        agg.correct_accusations += metrics.correct_accusations
        agg.false_accusations += metrics.false_accusations
        agg.grounded_accusations += metrics.grounded_accusations
        agg.ungrounded_accusations += metrics.ungrounded_accusations
        agg.directly_observed_accusations += metrics.directly_observed_accusations
        agg.testimony_derived_accusations += metrics.testimony_derived_accusations
        agg.unsupported_accusations += metrics.unsupported_accusations

        per_run_summary.append({
            "run_dir": str(run_dir),
            "tier3": metrics.to_dict(),
        })

        if batch:
            print(
                f"[ok] {rel}: "
                f"acc={metrics.total_accusations} "
                f"(obs {metrics.directly_observed_accusations} / "
                f"test {metrics.testimony_derived_accusations} / "
                f"unsup {metrics.unsupported_accusations})"
            )
        else:
            print(f"\nRun: {rel}")
            print("\nVerdict transitions (old -> new):")
            for k, v in sorted(verdict_diff.items(), key=lambda x: -x[1]):
                if v > 0:
                    print(f"  {k:<30} {v:>4}")
            print("\nAccusation basis (re-verified from cached claims):")
            _print_accusation_summary(metrics)

    # Recompute aggregate rates from the summed counts.
    t = agg.total_accusations
    if t > 0:
        agg.accusation_accuracy = agg.correct_accusations / t
        agg.unsupported_accusation_rate = agg.ungrounded_accusations / t
        agg.directly_observed_accusation_rate = agg.directly_observed_accusations / t
        agg.testimony_derived_accusation_rate = agg.testimony_derived_accusations / t
        agg.strictly_unsupported_accusation_rate = agg.unsupported_accusations / t

    print("\n" + "=" * 60)
    print(f"AGGREGATE over {processed} run(s)"
          + (f" ({failed} failed)" if failed else ""))
    print("=" * 60)
    print("\nAccusation basis (summed across runs):")
    _print_accusation_summary(agg)
    print("\nVerdict transitions (old -> new), summed:")
    for k, v in sorted(agg_verdict_diff.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  {k:<30} {v:>4}")

    # Write an aggregate summary next to the root for downstream analysis.
    summary_path = root / "reverify_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "root": str(root),
                "num_runs": processed,
                "num_failed": failed,
                "aggregate_tier3": agg.to_dict(),
                "aggregate_verdict_transitions": agg_verdict_diff,
                "per_run": per_run_summary,
            },
            f, indent=2, default=str,
        )
    print(f"\nWrote aggregate summary to {summary_path}")
    print("Per-run outputs: tier3_claims_reverified.jsonl + evaluation_reverified.json")


if __name__ == "__main__":
    main()
