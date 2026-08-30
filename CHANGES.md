# Changes

Changes made in response to the ARR May cycle reviews. Three additions:

1. A flag to run games in **text-only** mode (no images to the VLM), for a
   text-only vs. image+text ablation.
2. Support for **Claude Opus 4.7** as the Tier 3 extraction model (in addition
   to the default GPT-5.5).
3. A **three-way accusation taxonomy** — *directly-observed*,
   *testimony-derived*, *unsupported* — refining the previous binary
   grounded/ungrounded split.

---

## 1. Text-only ablation flag (`send_images`)

Adds a switch that controls whether each agent's rendered images (global map +
local view) are attached to the VLM request. The default is unchanged
(image + structured text). Setting the flag to `false` sends **only** the
structured text observation, enabling a clean text-only ablation.

**Usage**

```bash
# Default: image + text (unchanged behavior)
python scripts/run_game.py model=gpt5.5 seed=42

# Text-only ablation (no images sent to the VLM)
python scripts/run_game.py model=gpt5.5 seed=42 send_images=false
```

When `send_images=false` the run banner prints
`input: TEXT-ONLY (images disabled)`, and the frozen `config.yaml` snapshot
saved in the run directory records the flag, so ablation runs are
self-documenting.

**Behavioral notes**

- The engine still *renders* the god-view video and per-agent images as before;
  only the attachment of images to the model request is suppressed. This keeps
  videos/replays identical across the ablation.
- The agent **system prompt adapts** to the flag. With images off, the
  "YOUR VISION" section no longer promises "two images each tick" and instead
  tells the model all spatial information is provided as structured text
  (current room, adjacency + travel times, task locations, visible
  players/bodies). This avoids instructing the model to look for images that
  were never sent.
- Works for both homogeneous and heterogeneous experiments (the flag is applied
  to duck agents on reassignment too).

**Files changed**

- `configs/config.yaml` — new `send_images: true` runtime option (documented
  inline).
- `quack/agents/vlm_agent.py` — `VLMAgent.__init__` gains `send_images: bool = True`;
  `_collect_images()` returns `[]` when disabled; `on_game_start` forwards the
  flag to the system-prompt builder.
- `quack/agents/prompt_builder.py` — `build_system_prompt` gains
  `send_images: bool = True` and swaps the "YOUR VISION" section between the
  image and text-only descriptions.
- `scripts/run_game.py` — threads `cfg.send_images` into
  `create_agents_from_config` and `reassign_duck_agents`; prints the input mode
  in the startup banner.

---

## 2. Claude Opus 4.7 as a Tier 3 extraction model

The evaluation extraction model (previously GPT-5.5 only, routed through the
greatrouter proxy via LiteLLM) can now be **Claude Opus 4.7**. Claude models are
routed **directly to the Anthropic API** (as the game agents already are —
greatrouter heavily rate-limits Claude), with the required `max_tokens` set
automatically.

**Usage**

```bash
# Default extractor (unchanged)
python scripts/evaluate_game.py path/to/game.jsonl --tier3

# Claude Opus 4.7 as the extraction model
python scripts/evaluate_game.py path/to/game.jsonl --tier3 --model claude-opus-4-7

# Batch, Claude extractor
python scripts/evaluate_batch.py game_logs/homogeneous/gpt5.5/ --tier3 --model claude-opus-4-7
```

The model id `claude-opus-4-7` matches `configs/model/claude_opus4.7.yaml`.
LiteLLM's `anthropic/` provider prefix is applied automatically, so both
`claude-opus-4-7` and `anthropic/claude-opus-4-7` work.

**Key resolution**

The evaluation CLIs now pick the right credentials based on the extraction
model (mirroring `run_game.py`):

- Anthropic/Claude model → `$ANTHROPIC_API_KEY`, else `claude_key.txt`.
- OpenAI-compatible model (e.g. `gpt-5.5`) → `$QUACK_API_KEY`, else
  `api_key.txt`.
- An explicit `--api-key` always overrides.

For Claude the greatrouter `--base-url` is ignored (the request goes to
`api.anthropic.com`); no `--base-url` override is needed.

**Output files are namespaced per extraction model (no overwrites)**

To let two extraction models be evaluated over the same game log without
clobbering each other's results, non-default models write to model-suffixed
filenames. The default `gpt-5.5` keeps the historical, un-suffixed names for
backward compatibility.

| Extraction model | Per-game audit | Per-game summary | Batch condition / overall |
|---|---|---|---|
| `gpt-5.5` (default) | `tier3_claims.jsonl` | `evaluation.json` | `<condition>.json`, `summary.json` |
| `claude-opus-4-7` | `tier3_claims_claude-opus-4-7.jsonl` | `evaluation_claude-opus-4-7.json` | `<condition>_claude-opus-4-7.json`, `summary_claude-opus-4-7.json` |

An explicit `--output/-o` is always respected as-is (no suffix added). The
suffix is a sanitized model slug (any `/` in a LiteLLM provider prefix becomes
`_`).

**Reproducibility note**

The Tier 3 extraction cache key includes the model string, so switching the
extractor to `claude-opus-4-7` produces **new cache entries** and does not reuse
or overwrite the existing `gpt-5.5|...` cached extractions. The two extractors'
caches coexist in the same `tier3_extraction_cache.jsonl` (this file is shared
and appended-to, never rewritten).

**Files changed**

- `quack/evaluation/tier3_statement_verification.py` —
  new `is_anthropic_extraction_model()`; `_call_extraction_llm` is now
  provider-aware (adds the `anthropic/` prefix, sets `max_tokens`, and skips a
  non-Anthropic `base_url` for Claude models); new `DEFAULT_EXTRACTION_MODEL`
  and `extraction_output_suffix()` for per-model output filenames.
- `quack/evaluation/extraction_keys.py` (new) — shared
  `resolve_extraction_api_key()` used by both eval CLIs.
- `quack/evaluation/evaluator.py` — Tier 3 audit filename gets the model
  suffix.
- `scripts/evaluate_game.py`, `scripts/evaluate_batch.py` — use the shared key
  resolver; updated `--model` help text; auto-detected `evaluation*.json`,
  per-condition, and `summary*.json` filenames get the model suffix.

---

## 3. Three-way accusation taxonomy

Following the reviewer suggestion, accusations are now classified into three
buckets instead of the previous binary grounded/ungrounded split:

- **`directly_observed`** — the accuser personally had an observational basis
  for the accusation (visibility on the target during the round, co-location at
  a kill/body scene, or being the meeting caller). This is exactly the old
  "grounded" condition.
- **`testimony_derived`** — the accuser had **no** direct observation, but
  another player named or accused the target **earlier in the same meeting**
  (before the accuser first spoke). The accusation is plausibly derived from
  that testimony rather than fabricated.
- **`unsupported`** — neither direct observation nor prior testimony: the true
  "unsupported accusation" failure mode.

Precedence is `directly_observed` > `testimony_derived` > `unsupported`.

As the reviewer noted, this only re-runs the **verdict logic** — no new LLM
extraction is required. The classification is computed from cached structured
claims plus the game log, so it can be applied to already-evaluated runs
offline.

**How to regenerate metrics for existing runs (no LLM calls)**

Use the offline re-verifier, which now emits the three-way breakdown and works
on either a single run **or a whole setting directory** (batch):

```bash
# single run
python scripts/reverify_tier3_from_audit.py game_logs/homogeneous/gpt5.5/<run>/

# every run under a setting (recursively finds each tier3_claims.jsonl)
python scripts/reverify_tier3_from_audit.py game_logs/homogeneous/gpt5.5/
```

For each run it re-reads the cached `tier3_claims.jsonl`, re-runs verification,
and writes `tier3_claims_reverified.jsonl` + `evaluation_reverified.json`
alongside it. In batch mode it also prints and writes an **aggregate** summary
across all runs to `<root>/reverify_summary.json`:

```
AGGREGATE over 25 run(s)
Accusation basis (summed across runs):
  total accusations:         74
  directly-observed:         34 (45.9%)
  testimony-derived:         29 (39.2%)
  unsupported:               11 (14.9%)
```

The map is resolved per-run from each log (falling back to `--map-config`), and
individual run failures are reported without aborting the batch. A fresh
`--tier3` evaluation also emits the new fields directly.

*Caveat:* the audit's `structured_claim` does not persist multi-room `route`
arrays, so route-only location claims re-verify as `unverifiable` (they can't
be reconstructed without the original extraction). This is inherent to
reverify-from-audit and does not affect the accusation taxonomy (accusations
carry no room/route).

**New metrics** (in `Tier3Metrics` / `evaluation.json` → `tier3`)

- Counts: `directly_observed_accusations`, `testimony_derived_accusations`,
  `unsupported_accusations`.
- Rates: `directly_observed_accusation_rate`,
  `testimony_derived_accusation_rate`, `strictly_unsupported_accusation_rate`.
- Per-claim audit (`tier3_claims.jsonl` → `verification.evidence`): new
  `accusation_basis` field on every accusation.
- Per-player breakdown (`per_player_claims`): new
  `accusation_directly_observed` / `accusation_testimony_derived` /
  `accusation_unsupported` counts.

**Backward compatibility**

The legacy binary metrics are **unchanged** so prior numbers remain
reproducible:

- `grounded_accusations` == `directly_observed_accusations`.
- `ungrounded_accusations` == `testimony_derived_accusations` +
  `unsupported_accusations`.
- The legacy `unsupported_accusation_rate` still equals the ungrounded rate
  (= not directly observed). The new strict rate is
  `strictly_unsupported_accusation_rate`.

The new three-way rates are also added to the batch-aggregation field lists so
they appear (mean/std) in aggregated summaries.

**Files changed**

- `quack/evaluation/tier3_statement_verification.py` —
  `_classify_accusation_basis()`, `_accusation_supported_by_testimony()`,
  `_get_meetings_cached()`, and module-level `_mentions_player()`;
  `_verify_accusation` now records `accusation_basis`; `Tier3Metrics` gains the
  new count/rate fields and serializes them; `_compute_metrics` accumulates the
  three buckets (overall + per-player).
- `quack/evaluation/report.py` — prints the accusation-basis breakdown.
- `quack/evaluation/evaluator.py`, `scripts/evaluate_batch.py` — the new rates
  added to Tier 3 aggregation.
- `scripts/reverify_tier3_from_audit.py` — prints the three-way summary; now
  **batch-capable** (accepts a setting directory, re-verifies every run under
  it, resolves each run's map from its log, and writes a
  `reverify_summary.json` aggregate).
- `quack/evaluation/tier3_statement_verification.py` — `normalize_room_name`
  now tolerates `None`/empty input (returns `None` → the already-intended
  "unverifiable" path) instead of raising `AttributeError`. This hardens both
  the live pipeline and reverify against location claims that carry
  `room=None`.

---

## Testing

- Full existing suite passes (`python -m pytest tests/ -q` → 253 passed).
- Added ad-hoc verification that: the `send_images` flag toggles both the
  attached images and the system-prompt vision section; the Anthropic model
  detection/key resolution behave correctly; and the accusation classifier
  returns `directly_observed` / `testimony_derived` / `unsupported` in the
  expected cases (direct observation takes precedence; a prior speaker naming
  the target yields `testimony_derived`; otherwise `unsupported`).
