"""Rough token-cost analysis: game rollout vs Tier 3 extraction, for one setting.

Counts the *exact* number of LLM calls from the logs, measures the fixed
system prompt and the (exactly reconstructable) extraction prompts with
tiktoken, and applies clearly-labeled estimates for the parts that aren't
recorded (per-tick observation text, rendered images, output lengths).

Usage:
    python scripts/analyze_token_cost.py game_logs/homogeneous/gpt5.5/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quack.agents.prompt_builder import build_system_prompt
from quack.evaluation.tier3_statement_verification import (
    EXTRACTION_PROMPT, EXTRACTION_PROMPT_VERSION,
)

ENC = tiktoken.get_encoding("cl100k_base")


def ntok(s: str) -> int:
    return len(ENC.encode(s or ""))


# ---- Labeled estimates for parts not recorded in the logs ----------------
# Per game VLM call, image inputs (2 rendered PNGs: global map + local view).
# ~765 tok/image under OpenAI high-detail tiling for ~1024px diagrams.
IMG_TOKENS_PER_CALL = 1530
# Per-call structured observation / user text (rough averages).
USER_TOK_ACTION = 260
USER_TOK_SPEAK = 300
USER_TOK_VOTE = 200
# Output tokens per call type (rough).
OUT_TOK_ACTION = 15
OUT_TOK_SPEAK = 120
OUT_TOK_VOTE = 10


def analyze_run(run_dir: Path) -> dict | None:
    game_log = run_dir / "game.jsonl"
    if not game_log.exists():
        return None
    events = [json.loads(l) for l in open(game_log) if l.strip()]

    players: list[str] = []
    num_players = 0
    for e in events:
        if e.get("event_type") == "game_started":
            players = e["data"].get("players", [])
            num_players = e["data"].get("config", {}).get("num_players", len(players))
            break
    id_to_name = {f"player_{i}": n for i, n in enumerate(players)}

    # Deaths: permanent, keyed by the tick they occur.
    death_ticks: list[int] = []
    for e in events:
        if e.get("event_type") in ("player_killed", "player_ejected"):
            death_ticks.append(e.get("tick", 0))

    def alive_at(tick: int) -> int:
        # Alive when the decision for `tick` is made = players not yet dead
        # before this tick.
        return num_players - sum(1 for dt in death_ticks if dt < tick)

    free_roam_ticks = sorted(e["tick"] for e in events if e.get("event_type") == "tick_start")
    meetings = [e for e in events if e.get("event_type") in ("body_reported", "meeting_called")]
    disc_msgs = [e for e in events if e.get("event_type") == "discussion_message"]

    # --- Game VLM calls -----------------------------------------------------
    # choose_action: one per alive player per free-roam tick (upper bound;
    # players mid-transit may not always be prompted).
    action_calls = sum(alive_at(t) for t in free_roam_ticks)
    speak_calls = len(disc_msgs)
    # vote: one per alive player per meeting/voting phase.
    vote_calls = sum(alive_at(m["tick"]) for m in meetings)
    game_calls = action_calls + speak_calls + vote_calls

    # System prompt (fixed per game per role). Weight by role mix.
    obj = "Complete all your assigned tasks and help identify and vote out the Ducks."
    sys_goose = ntok(build_system_prompt(
        "Alice", "Goose", "goose", obj,
        total_geese=num_players - 1, total_ducks=1,
        all_players=players, send_images=True,
    ))
    sys_duck = ntok(build_system_prompt(
        "Bob", "Duck", "duck", obj,
        total_geese=num_players - 1, total_ducks=1,
        teammates=[], all_players=players, send_images=True,
    ))
    # 1 duck of num_players → weighted average system prompt size.
    sys_avg = (sys_goose * (num_players - 1) + sys_duck) / num_players

    # Game input/output token totals (system + user text + images).
    game_in = (
        action_calls * (sys_avg + USER_TOK_ACTION + IMG_TOKENS_PER_CALL)
        + speak_calls * (sys_avg + USER_TOK_SPEAK + IMG_TOKENS_PER_CALL)
        + vote_calls * (sys_avg + USER_TOK_VOTE + IMG_TOKENS_PER_CALL)
    )
    game_out = (
        action_calls * OUT_TOK_ACTION
        + speak_calls * OUT_TOK_SPEAK
        + vote_calls * OUT_TOK_VOTE
    )
    game_img = (action_calls + speak_calls + vote_calls) * IMG_TOKENS_PER_CALL

    # --- Extraction (Tier 3) — measured exactly ----------------------------
    # Input: reconstruct the exact extraction prompt per discussion message.
    # Meeting tick per message: nearest preceding meeting.
    meeting_ticks = sorted(m["tick"] for m in meetings)

    def meeting_tick_for(t: int) -> int:
        prev = [mt for mt in meeting_ticks if mt <= t]
        return prev[-1] if prev else (meeting_ticks[0] if meeting_ticks else t)

    extr_in = 0
    for e in disc_msgs:
        d = e["data"]
        speaker = id_to_name.get(d.get("player_id", ""), d.get("player_id", ""))
        mt = meeting_tick_for(e.get("tick", 0))
        prompt = EXTRACTION_PROMPT.format(
            speaker_name=speaker, meeting_tick=mt,
            player_names=", ".join(players), message=d.get("message", ""),
        )
        extr_in += ntok(prompt)

    # Output: token count of the cached claim JSON for this run's gpt-5.5
    # entries. NOTE: not every run has had extraction run, so cached entries
    # can be < number of messages; we report cached figures separately and
    # normalize to all messages at the aggregate level.
    cached_out = 0
    cached_entries = 0
    cache = run_dir / "tier3_extraction_cache.jsonl"
    if cache.exists():
        for l in open(cache):
            l = l.strip()
            if not l:
                continue
            entry = json.loads(l)
            key = entry.get("key", "")
            if key.startswith(f"gpt-5.5|{EXTRACTION_PROMPT_VERSION}|"):
                cached_entries += 1
                cached_out += ntok(json.dumps(entry.get("claims", [])))

    return {
        "run": run_dir.name,
        "num_players": num_players,
        "end_tick": free_roam_ticks[-1] if free_roam_ticks else 0,
        "free_roam_ticks": len(free_roam_ticks),
        "meetings": len(meetings),
        "action_calls": action_calls,
        "speak_calls": speak_calls,
        "vote_calls": vote_calls,
        "game_calls": game_calls,
        "sys_goose_tok": sys_goose,
        "sys_duck_tok": sys_duck,
        "game_in_tok": round(game_in),
        "game_out_tok": round(game_out),
        "game_img_tok": round(game_img),
        "game_total_tok": round(game_in + game_out),
        "extr_msgs": len(disc_msgs),
        "extr_in_tok": extr_in,
        "cached_entries": cached_entries,
        "cached_out_tok": cached_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("setting_dir")
    args = ap.parse_args()
    root = Path(args.setting_dir)

    runs = sorted(p.parent for p in root.rglob("game.jsonl"))
    rows = [r for r in (analyze_run(rd) for rd in runs) if r]
    if not rows:
        print("no runs found")
        return

    n = len(rows)

    def s(key):
        return sum(r[key] for r in rows)

    def avg(key):
        return s(key) / n

    # Normalize extraction OUTPUT to all messages: not every run had
    # extraction run, so estimate mean output tokens/message from the runs
    # that do have cache, then apply to every discussion message.
    out_per_msg = (s("cached_out_tok") / s("cached_entries")) if s("cached_entries") else 0.0
    total_extr_msgs = s("extr_msgs")
    total_extr_in = s("extr_in_tok")
    total_extr_out = out_per_msg * total_extr_msgs
    total_extr = total_extr_in + total_extr_out

    total_game = s("game_total_tok")

    print(f"Setting: {root}")
    print(f"Runs analyzed: {n}")
    print(f"System prompt (fixed/game): goose={rows[0]['sys_goose_tok']} tok, "
          f"duck={rows[0]['sys_duck_tok']} tok")
    print(f"Extraction cache coverage: {s('cached_entries')}/{total_extr_msgs} messages "
          f"({100*s('cached_entries')/total_extr_msgs:.0f}%); mean {out_per_msg:.0f} out-tok/msg")
    print()
    print("Per game (averaged across runs):")
    print(f"  end tick / free-roam ticks:  {avg('end_tick'):.1f} / {avg('free_roam_ticks'):.1f}")
    print(f"  GAME  VLM calls:  {avg('game_calls'):7.1f}  "
          f"(action {avg('action_calls'):.1f}, speak {avg('speak_calls'):.1f}, vote {avg('vote_calls'):.1f})")
    print(f"  EXTR  LLM calls:  {avg('extr_msgs'):7.1f}  (= discussion messages)")
    print()
    print(f"  GAME  tokens:  in {avg('game_in_tok'):9.0f} | out {avg('game_out_tok'):7.0f} "
          f"| total {total_game/n:9.0f}   (of which images ~{avg('game_img_tok'):.0f})")
    print(f"  EXTR  tokens:  in {total_extr_in/n:9.0f} | out {total_extr_out/n:7.0f} "
          f"| total {total_extr/n:9.0f}")
    ratio = total_game / total_extr if total_extr else 0
    print(f"  GAME / EXTR total-token ratio: {ratio:.1f}x")
    print()
    print(f"Whole setting ({n} games) totals:")
    print(f"  GAME  total tokens: {total_game:14,.0f}")
    print(f"  EXTR  total tokens: {total_extr:14,.0f}")
    tot = total_game + total_extr
    print(f"  Combined:           {tot:14,.0f}")
    print(f"  Extraction share:   {100*total_extr/tot:.1f}%")

    out = root / "token_cost_analysis.json"
    with open(out, "w") as f:
        json.dump({
            "setting": str(root), "num_runs": n,
            "assumptions": {
                "img_tokens_per_call": IMG_TOKENS_PER_CALL,
                "user_tok_action": USER_TOK_ACTION,
                "user_tok_speak": USER_TOK_SPEAK,
                "user_tok_vote": USER_TOK_VOTE,
                "out_tok_action": OUT_TOK_ACTION,
                "out_tok_speak": OUT_TOK_SPEAK,
                "out_tok_vote": OUT_TOK_VOTE,
            },
            "totals": {
                "game_total_tok": total_game,
                "extr_in_tok": total_extr_in,
                "extr_out_tok_est": total_extr_out,
                "extr_total_tok": total_extr,
                "extr_out_per_msg": out_per_msg,
                "game_over_extr_ratio": ratio,
                "extraction_share_pct": 100 * total_extr / tot,
            },
            "per_run": rows,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
