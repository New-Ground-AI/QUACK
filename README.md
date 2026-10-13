[![arXiv](https://img.shields.io/badge/arXiv-2605.27068-b31b1b)](https://arxiv.org/abs/2605.27068)
[![Hugging Face Paper](https://img.shields.io/badge/🤗%20Paper-2605.27068-yellow)](https://huggingface.co/papers/2605.27068)

# QUACK🦆: Questioning, Understanding, and Auditing Collaborative Knowledge

We are happy to announce that this project is accepted by EMNLP 2026! This repository contains the official implementation for the paper.

See the [project overview](https://new-ground-ai.github.io/projects/quack/) and explore related research from [New Ground AI](https://new-ground-ai.github.io/).

### A Multimodal Social Deduction Environment for Vision-Language Model Agents

Recent work has increasingly used social deduction games to study reasoning, deception, coordination, and theory of mind in Large Language Model (LLM) agents. However, existing environments are typically evaluated through game outcomes such as win rates, and most remain limited to text-only interaction. As a result, it is difficult to determine whether an agent's reasoning is grounded in what it actually perceived and did, or to systematically identify the failure modes underlying its behavior.

To address this gap, **QUACK** is an open-source environment and evaluation framework for auditing multimodal social reasoning in Vision-Language Model (VLM) agents. Agents navigate configurable graph-based maps, observe rendered global and local views, complete location-bound tasks, communicate through free-form discussion, and vote under hidden-role adversarial incentives. QUACK evaluates agents at three levels: **game outcomes, behavioral trajectories, and utterance-level consistency**. Its evaluation framework reconstructs agent trajectories and verifies dialogue against engine-level logs, enabling automatic identification of failures such as **spatial hallucination, unsupported accusation, deception collapse, and language-action inconsistency**. 

![Alt Text](assets/video.gif)

---

## Why This Project?

Most existing social deduction agent environments (Werewolf, Mafia, Avalon) are text-only — agents read and write natural language. This limits the analysis to LLM capabilities alone and to coarse game-level outcomes.

QUACK introduces a **spatial dimension** and a **claim-verification layer**: agents navigate a discrete map, have limited local vision, receive rendered map images as input, and every utterance they produce during meetings is automatically grounded against engine-level logs. This makes it possible to audit VLM agents along four dimensions:

1. **Read and interpret visual maps** — understand room layouts, task markers, and visible bystanders.
2. **Reason spatially** — plan movement routes over weighted corridors, track encounters, and remember witnessed traffic.
3. **Deduce socially** — identify impostors through discussion and voting under partial observability.
4. **Stay grounded** — utterances must be consistent with the agent's actual trajectory; lies, hallucinations, and unsupported accusations are flagged automatically.

---

## Quick Start

### Installation

```bash
cd QUACK

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Run a Game (Random Agents — no API key needed)

```bash
# Basic game with random agents (no api_key.txt required)
python scripts/run_game.py

# With a fixed seed for reproducibility
python scripts/run_game.py seed=42

# Disable video generation (faster)
python scripts/run_game.py video=false seed=42
```

### Run a Game (VLM Agents)

```bash
# Greatrouter key (used by GPT-5.5 and Gemini 3.1 Pro Preview)
echo "sk-your-greatrouter-key" > api_key.txt

# Official Anthropic key (used by Claude Opus 4.7 — bypasses greatrouter
# rate limits, hits api.anthropic.com directly)
echo "sk-ant-your-anthropic-key" > claude_key.txt

# Run with GPT-5.5 (default model)
python scripts/run_game.py seed=42

# Run with Claude Opus 4.7  (reads claude_key.txt automatically)
python scripts/run_game.py model=claude_opus4.7 seed=42

# Run with Gemini 3.1 Pro Preview (uses streaming automatically)
python scripts/run_game.py model=gemini3.1pro seed=42

# VLM agents with Simplified Chinese speech (简体中文)
python scripts/run_game.py speak_chinese=true seed=42

# Custom API endpoint (e.g. domestic China backup node — only affects the
# greatrouter-backed models; Claude always hits api.anthropic.com)
python scripts/run_game.py base_url=https://endpoint.wendalog.com
```

### Heterogeneous Experiments

```bash
# GPT-5.5 geese vs Claude Opus 4.7 duck
python scripts/run_game.py experiment=heterogeneous model=gpt5.5 experiment.duck_model=claude_opus4.7 seed=42

# Claude geese vs Gemini duck
python scripts/run_game.py experiment=heterogeneous model=claude_opus4.7 experiment.duck_model=gemini3.1pro seed=42
```

### Batch Runs

Three shell scripts automate large-scale experiments. All support `-h` for full usage.

#### Homogeneous Batch (`batch_homogeneous.sh`)

```bash
# All 3 models × 30 games each (default)
./scripts/batch_homogeneous.sh

# Single model, 10 games
./scripts/batch_homogeneous.sh -m gpt5.5 -n 10

# Two models, 5 games each, starting from seed 11
./scripts/batch_homogeneous.sh -m gpt5.5,claude_opus4.7 -n 5 -s 11

# Skip video generation for faster runs
./scripts/batch_homogeneous.sh -m gpt5.5 -n 10 -- video=false
```

#### Heterogeneous Batch (`batch_heterogeneous.sh`)

```bash
# All cross-model pairs (6 pairs) × 30 games each
./scripts/batch_heterogeneous.sh

# Specific goose/duck pair, 10 games
./scripts/batch_heterogeneous.sh -g gpt5.5 -d claude_opus4.7 -n 10

# GPT-5.5 geese vs every other model as duck, 5 games each
./scripts/batch_heterogeneous.sh -g gpt5.5 -d all -n 5
```

#### Full Experiment Suite (`batch_full_experiment.sh`)

```bash
# Full matrix: 3 homogeneous + 6 heterogeneous = 9 conditions × 30 games = 270 games
./scripts/batch_full_experiment.sh

# Quick test run: 9 conditions × 5 games = 45 games
./scripts/batch_full_experiment.sh -n 5

# Run everything + auto-evaluate Tier 1/2 at the end (add --tier3 to evaluate_batch.py separately for Tier 3)
./scripts/batch_full_experiment.sh -n 10 --eval
```

#### Manual loop (alternative)

```bash
# Run 30 games with Gemini 3.1 Pro Preview
for seed in $(seq 1 30); do
    python scripts/run_game.py model=gemini3.1pro seed=$seed
done

# Run 30 heterogeneous games
for seed in $(seq 1 30); do
    python scripts/run_game.py experiment=heterogeneous model=gpt5.5 experiment.duck_model=claude_opus4.7 seed=$seed
done
```

### Replay a Game Log

Re-render god-view frames and `video.mp4` from a saved `game.jsonl` (use this after rendering logic changes, or when videos are missing). Output goes into the run directory by default:

```bash
RUN=game_logs/homogeneous/gpt5.5/20260520_230743_seed1

# Frames only → <run>/renders/god_view/
python scripts/replay_game.py "$RUN/game.jsonl" \
  -o "$RUN/renders/god_view"

# Frames + video (use --fps 1 to match run_game.py default)
python scripts/replay_game.py "$RUN/game.jsonl" \
  -o "$RUN/renders/god_view" \
  --video "$RUN/video.mp4" \
  --fps 1
```

#### Batch replay (all runs)

`replay_all.sh` re-renders every `game.jsonl` under `game_logs/` (270 games = 9 conditions × 30 seeds). It reads all paths into an array first so ffmpeg cannot corrupt the find pipe (a known bash stdin issue).

```bash
# All runs, 1 fps (matches run_game.py video_fps default)
./scripts/replay_all.sh

# One condition only
./scripts/replay_all.sh game_logs/homogeneous/gpt5.5/

# Custom fps
./scripts/replay_all.sh -f 2
```

**Note:** `game.jsonl` must contain `initial_state` in the `game_started` event (all current logs do). Map paths in old logs may be absolute; normalize them first (see [Log maintenance](#log-maintenance) below).

### Generate Videos from Existing Frames

`generate_videos.sh` **only re-encodes** existing `renders/god_view/frame_*.png` into `video.mp4`. It does **not** re-render from `game.jsonl`. To regenerate frames from logs, use `replay_game.py` or `replay_all.sh` above.

```bash
# Encode video.mp4 for runs that have frames but no video yet (1 fps)
./scripts/generate_videos.sh

# Only for a specific model
./scripts/generate_videos.sh game_logs/homogeneous/gpt5.5/

# Custom frame rate
./scripts/generate_videos.sh -f 2

# Force re-encode all videos (frames must already exist)
./scripts/generate_videos.sh --force
```

### Evaluate Games

Tier 1 and Tier 2 run from log data alone (no API key). Tier 3 uses an LLM to extract meeting claims and verify them against the reconstructed timeline.

#### Single game

```bash
# Tier 1 + Tier 2 — auto-saves evaluation.json next to game.jsonl
python scripts/evaluate_game.py game_logs/homogeneous/gpt5.5/20260520_230743_seed1/game.jsonl

# Tier 3 — reads api_key.txt automatically if --api-key is omitted
python scripts/evaluate_game.py game_logs/homogeneous/gpt5.5/20260520_230743_seed1/game.jsonl --tier3

# Skip the per-claim audit file (default: write tier3_claims.jsonl)
python scripts/evaluate_game.py ... --tier3 --no-tier3-audit
```

With `--tier3`, each run also produces:
- `tier3_claims.jsonl` — one JSON object per extracted claim, with verdict, evidence, and `ground_truth_events`
- `tier3_extraction_cache.jsonl` — cached LLM extraction output (re-runs reuse cache; only verification re-runs)

#### One condition or all logs

```bash
# One homogeneous model (30 games)
python scripts/evaluate_batch.py game_logs/homogeneous/gpt5.5/ --tier3

# One heterogeneous pair
python scripts/evaluate_batch.py game_logs/heterogeneous/geese_gpt5.5_duck_claude_opus4.7/ --tier3

# Everything under game_logs/ (270 games across 9 conditions)
python scripts/evaluate_batch.py game_logs/ --tier3
```

Aggregated JSON is written to `<search_root>/evaluation/{condition}.json` and `summary.json`. Per-game `evaluation.json` is saved alongside each `game.jsonl`.

#### Full experiment matrix (9 conditions × 30 games)

```bash
# Run games only
./scripts/batch_full_experiment.sh

# Run games + Tier 1/2 evaluation at the end (no Tier 3 unless you add --tier3 manually)
./scripts/batch_full_experiment.sh --eval

# Evaluate all 9 conditions with Tier 3 after games are done
python scripts/evaluate_batch.py game_logs/homogeneous/gpt5.5/ --tier3
python scripts/evaluate_batch.py game_logs/homogeneous/claude_opus4.7/ --tier3
python scripts/evaluate_batch.py game_logs/homogeneous/gemini3.1pro/ --tier3
# ... repeat for each heterogeneous condition, or:
python scripts/evaluate_batch.py game_logs/ --tier3
```

To merge per-condition batch outputs into one file (e.g. all 270 games):

```bash
python scripts/merge_batch_results.py \
  game_logs/all_evaluation/homogeneous_gpt5.5.json \
  game_logs/all_evaluation/homogeneous_claude_opus4.7.json \
  ... \
  -o game_logs/all_evaluation/summary_270.json
```

#### Tier 3 validation & re-verification

```bash
# Recompute Tier 3 metrics from tier3_claims.jsonl; assert they match evaluation.json
python scripts/validate_tier3_audit.py --check game_logs/homogeneous/gpt5.5/20260520_230743_seed1/

# Re-run verification only (skip LLM extraction) on an existing audit file
python scripts/reverify_tier3_from_audit.py game_logs/homogeneous/gpt5.5/20260520_230743_seed1/
```

### Log maintenance

Old logs may store absolute `map` paths from different machines. Normalize them to the repo-relative path before replay/evaluation on a new checkout:

```bash
python scripts/normalize_map_paths.py          # dry-run
python scripts/normalize_map_paths.py --write  # apply to all game.jsonl
```

---

## Configuration (Hydra)

QUACK uses [Hydra](https://hydra.cc/) for hierarchical configuration management. All settings are composable YAML files under `configs/`:

```
configs/
├── config.yaml                # Main entry point (defaults + runtime options)
├── game/
│   └── default.yaml           # Game rules (num_players, max_ticks, etc.)
├── map/
│   └── simple_ship.yaml       # Map definition (rooms, corridors, tasks)
├── model/
│   ├── gpt5.5.yaml            # GPT-5.5 (default)
│   ├── claude_opus4.7.yaml    # Claude Opus 4.7
│   └── gemini3.1pro.yaml      # Gemini 3.1 Pro Preview (streaming)
└── experiment/
    ├── homogeneous.yaml       # All players use the same model
    └── heterogeneous.yaml     # Geese use one model, duck uses another
```

Override any setting on the command line using Hydra's `key=value` syntax:

```bash
# Override seed and model
python scripts/run_game.py model=claude_opus4.7 seed=42

# Override game settings
python scripts/run_game.py game.max_ticks=100 game.num_ducks=2

# Override nested settings
python scripts/run_game.py game.kill.cooldown_ticks=3 game.meeting.max_discussion_rounds=3

# Disable video and god view
python scripts/run_game.py video=false god_view=false
```

---

## Supported Models

QUACK currently ships with three frontier VLMs. GPT-5.5 and Gemini 3.1 Pro Preview are reached through the OpenAI-compatible greatrouter proxy (`https://endpoint.greatrouter.com` by default; `https://endpoint.wendalog.com` is available as a domestic-China backup node). Claude Opus 4.7 talks to the official Anthropic API directly because the proxy's rate limits were too aggressive for Opus:

| Config Name | Display Name | API Model ID | Provider | Endpoint | Streaming | Notes |
|---|---|---|---|---|---|---|
| `gpt5.5` | GPT-5.5 | `gpt-5.5` | `openai` (greatrouter) | `api_key.txt` / `base_url` | No | Default model |
| `claude_opus4.7` | Claude Opus 4.7 | `claude-opus-4-7` | `anthropic` (official) | `claude_key.txt` / `api.anthropic.com` | No | Bypasses greatrouter to avoid Opus throttling; requires `max_tokens` |
| `gemini3.1pro` | Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | `openai` (greatrouter) | `api_key.txt` / `base_url` | **Yes** | Requires streaming to avoid timeout; do **not** set `max_tokens` |

The `VLMAgent` switches SDKs based on the `provider` field in the model config (`openai` or `anthropic`). Both providers share the same internal message format (OpenAI-style `image_url` + `text` parts) — Anthropic conversion (pulling out `system`, rewriting `image_url` → typed `image` blocks with base64 source) happens transparently in the agent. Streaming behavior is controlled per-model via `requires_stream`.

### Adding a New Model

Create a YAML file at `configs/model/<name>.yaml`:

```yaml
name: "my_model"
display_name: "My Model"
model_id: "my-model-api-id"
provider: "openai"        # or "anthropic"
temperature: null         # omit field for frontier models that reject custom temperature
requires_stream: false
# Anthropic provider only — Anthropic requires max_tokens on every call:
# max_tokens: 4096
```

Then run: `python scripts/run_game.py model=my_model`

---

## Output Structure

Each experiment run produces a self-contained directory with everything needed for analysis and replay:

```
game_logs/
├── homogeneous/
│   ├── gpt5.5/
│   │   ├── 20260308_143022_seed42/
│   │   │   ├── game.jsonl           # Structured event log
│   │   │   ├── config.yaml          # Frozen Hydra config snapshot
│   │   │   ├── evaluation.json      # Evaluation results (after running evaluate)
│   │   │   ├── tier3_claims.jsonl   # Per-claim Tier 3 audit (with --tier3)
│   │   │   ├── tier3_extraction_cache.jsonl  # Cached LLM extractions
│   │   │   ├── renders/
│   │   │   │   └── god_view/
│   │   │   │       ├── frame_0001.png
│   │   │   │       ├── frame_0002.png
│   │   │   │       └── ...
│   │   │   └── video.mp4            # Auto-generated replay video
│   │   ├── 20260308_143522_seed43/
│   │   │   └── ...
│   │   └── ...
│   ├── claude_opus4.7/
│   └── gemini3.1pro/
├── heterogeneous/
│   ├── geese_gpt5.5_duck_claude_opus4.7/
│   │   └── 20260308_150000_seed42/
│   │       └── ...
│   └── ...
└── evaluation/                      # Aggregated batch-eval output (under search root)
    ├── homogeneous_gpt5.5.json
    ├── heterogeneous_geese_gpt5.5_duck_claude_opus4.7.json
    └── summary.json
```

You may also keep cross-condition summaries in a separate folder (e.g. `game_logs/all_evaluation/`) for paper/analysis; use `merge_batch_results.py` to combine them.

### Naming Convention

- **Homogeneous:** `game_logs/homogeneous/{model_name}/{timestamp}_seed{N}/`
- **Heterogeneous:** `game_logs/heterogeneous/geese_{goose_model}_duck_{duck_model}/{timestamp}_seed{N}/`
- **Random seed:** `{timestamp}_random` when no seed is specified.

---

## Understanding the Game

### Roles

- **Goose** — Crew members. Complete tasks to win. Can report bodies and call emergency meetings.
- **Duck** — Impostors. Kill Geese to gain voting majority. Blend in by pretending to do tasks. Can report bodies strategically.

### Win Conditions

| Team | How to Win |
|------|-----------|
| Goose | Complete all tasks **OR** eject all Ducks via voting |
| Duck | Kill enough Geese to reach voting majority (Ducks >= Geese alive) |
| Either | If max ticks reached, Geese win by default |

### Game Phases

```
FREE_ROAM ──→ MEETING_CALLED ──→ DISCUSSION ──→ VOTING ──→ EJECTION ──→ FREE_ROAM
    │              (body report or                                           │
    │               emergency bell)                                          │
    └────────────────────────────────────────────────────────────────────────┘
    │
    └─── (win condition met) ──→ GAME_OVER
```

1. **Free Roam** — Each tick, every alive player chooses an action:
   - `move(room_name)` — walk to an adjacent room (travel time varies by corridor weight)
   - `do_task()` — work on a task in the current room (takes multiple ticks)
   - `report()` — report a dead body in the room (triggers meeting)
   - `call_meeting()` — press the emergency button (cafeteria only, limited uses)
   - `kill(target_id)` — *Duck only* — kill a player in the same room
   - `wait()` — do nothing
   - **Optional chat** — with any action, an agent may append `| say(message)` to speak to players in the **same room** during Free Roam (e.g. `move(weapons) | say(I saw a body in storage)`).

2. **Discussion** — The player who reported the body or pressed the emergency button speaks **first**. Remaining players speak in random order. Up to 2 rounds (each player speaks twice max). Body location is only revealed by the reporter's speech — Ducks may lie about it; the God view shows the actual locations so observers can spot lies.

3. **Voting** — Each player votes to eject someone or skips. Majority wins; ties skip.

4. **Ejection** — The ejected player is eliminated. All living players are randomly respawned to new rooms. All bodies are cleared.

### Key Mechanics

- **Caller speaks first**: On body report or emergency meeting, the reporter/caller speaks first. Other players follow in random order.
- **Body location from reporter only**: The meeting reason does not reveal body location — only the first speaker (the reporter) describes it. Ducks can lie about location; God view displays actual locations for observers.
- **Free-roam chat**: During Free Roam, agents may attach `| say(message)` to any action to broadcast a short message to players in the **same room only**. Messages are not heard by players in other rooms or in corridors.
- **Weighted corridors**: Moving between rooms takes a variable number of ticks (1-3). Agents must plan routes efficiently.
- **Random spawns**: Players start in random rooms and are re-randomized after each meeting.
- **Emergency bell limit**: The total number of emergency meetings (bell pulls) per game equals the number of Ducks. Body reports are unlimited.
- **Kill cooldown**: Ducks must wait a cooldown (default: 5 ticks) between kills, with an initial delay (default: 5 ticks) before the first kill.
- **Same-room vision only**: Players can only see other players in the same room. Players in corridors can see others traveling the same corridor. No adjacent-room vision.

### Map

The game uses a **10-room ship map** with weighted corridors:

```
    oxygen ──(2)── cafeteria ──(2)── weapons
      │               │    ╲           │
     (1)             (1)    (2)       (1)
      │               │       ╲        │
  upper_engine ─(2)─ medbay ─(1)─ electrical ─(2)─ security
      │               │                              │
     (2)             (2)                             (2)
      │               │                              │
  lower_engine ─(2)─ storage ─────(3)────── navigation
```

Numbers in parentheses = travel ticks. Each room has themed pixel-art decorations (cafeteria has tables, medbay has beds, etc.) and unique task locations.

---

## VLM Agent Architecture

Each VLM agent (powered by a configurable frontier model — GPT-5.5 / Gemini 3.1 Pro Preview via the OpenAI-compatible greatrouter proxy, or Claude Opus 4.7 via the official Anthropic API) has:

### Vision Input (2 images per tick)

1. **Global map** — Shows the full ship layout with all rooms and corridors. The agent's task locations are marked with their color. **No other players are visible** on this map — only the room structure and the agent's own position.

2. **Local view** — A zoomed-in view of the agent's current room or corridor position. Shows players and bodies the agent can actually see right now (same-room only when in a room; same-corridor only when in transit).

### Memory System

Each agent maintains structured memory across the game:
- **Tick history**: room or corridor position, action taken, players seen, bodies found, free-roam chats heard
- **Encounter log**: who was seen, where, when
- **Meeting history**: what was discussed, who was ejected
- **Route description**: path taken since last meeting (for discussion)

### Strategy Prompts

Agents receive role-specific strategy guides embedded in their system prompt:

- **Goose strategy**: Task prioritization, buddy system, early vs late speaker tactics, evidence-based voting
- **Duck strategy**: Target isolation, kill timing, alibi building, teammate protection, strategic body reporting, discussion deflection

### Streaming Support

Some models (e.g. Gemini 3.1 Pro Preview) require streaming to avoid timeouts. This is configured per-model via `requires_stream: true` in the model config. The VLM agent automatically uses streaming when configured and never sets `max_tokens` (which can cause empty responses on the streaming endpoints).

### Rate Limiting & Retries

API calls are globally rate-limited (1 s minimum between calls). Transient failures (rate limits, timeouts, quota errors) trigger automatic retry with exponential backoff (up to 4 attempts). OpenAI and Anthropic SDK retry logs are suppressed.

---

## Rendering & Visualization

QUACK renders using **pixel-art style character sprites** with themed room interiors.

### Per-Player Global Map

Each agent sees a global map with:
- All rooms and corridors visible (full ship layout)
- Agent's own task locations marked with their color
- No other players shown — agents rely on local view for spatial awareness
- HUD with tick, phase, task progress, alive count

### Per-Player Local View

A zoomed view showing only the agent's current room:
- Players and bodies in the same room
- Adjacent room connections
- Used by VLM agents for spatial decision-making

### God View (Omniscient Observer)

An all-seeing overhead view for human observers and debugging:
- **All players** shown with pixel sprites and role labels (Goose/Duck)
- **Vision halos** — colored overlay showing each player's current vision
- **Action annotations** — last action per player, color-coded (gray=move, green=task, red=kill)
- **Right panel** — player roster, event log (includes kills, meetings, and free-roam chat lines)
- **Per-player POV row** — single horizontal row of every player's first-person local view below the main map (keeps video landscape)
- **Meeting frames** (1280x720) — dedicated frames for meeting calls, speeches, and vote results:
  - Two-panel layout: past speeches on left, current speaker's full message on right (readable fonts)
  - **Actual body locations** banner during discussion — shows the truth so you can see if the reporter (e.g. a Duck) is lying
- **Free-roam chat bubbles** — when players use `say(...)` during Free Roam, their latest message in that tick is rendered as a small text bubble under their character.

### Automatic Video Generation

When `video=true` (default) and `god_view=true` (default), the game automatically stitches god-view frames into an MP4 video using ffmpeg at `video_fps` (default **1**). If ffmpeg is not installed, a warning is logged and the video step is skipped (no crash).

```bash
# Manual stitching (if needed)
ffmpeg -framerate 1 -i renders/god_view/frame_%04d.png -c:v libx264 -pix_fmt yuv420p \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" video.mp4

# Or use the replay script (writes into the run directory)
python scripts/replay_game.py game_logs/homogeneous/gpt5.5/20260520_230743_seed1/game.jsonl \
  -o game_logs/homogeneous/gpt5.5/20260520_230743_seed1/renders/god_view \
  --video game_logs/homogeneous/gpt5.5/20260520_230743_seed1/video.mp4 \
  --fps 1
```

---

## Game Logs

Every game run saves a structured JSONL log to its experiment directory.

### Log Format

Each line is a JSON object:

```json
{"timestamp": 1772148671.23, "event_type": "player_moved", "tick": 5, "data": {"player_id": "player_0", "from": "cafeteria", "to": "medbay"}}
```

The `game_started` event contains the full initial state (roles, rooms, tasks) enabling complete game replay:

```json
{"event_type": "game_started", "tick": 0, "data": {"players": [...], "config": {...}, "initial_state": {"player_0": {"name": "Alice", "role": "Goose", "team": "goose", "room": "medbay", "tasks": [...]}, ...}}}
```

### Event Types

| Event | Key Data Fields |
|-------|----------------|
| `game_started` | `players`, `config`, `initial_state` |
| `tick_start` / `tick_end` | `tick` |
| `player_moved` | `player_id`, `from`, `to`, `ticks_remaining` |
| `player_killed` | `killer_id`, `target_id`, `room` |
| `body_reported` | `caller`, `reason`, `bodies` (actual room/victim for replay) |
| `meeting_called` | `caller`, `reason` |
| `discussion_message` | `player_id`, `message` |
| `vote_cast` | `voter`, `target` |
| `player_ejected` | `player_id`, `name`, `role`, `team` |
| `vote_skipped` | `reason` |
| `task_progress` | `player_id`, `task_name`, `ticks_done` |
| `task_completed` | `player_id`, `task_name`, `room` |
| `free_roam_chat` | `player_id`, `name`, `room`, `message` |
| `players_respawned` | `players` (post-ejection random respawn positions) |
| `phase_changed` | `phase` |
| `game_over` | `winner`, `reason` |

---

## Evaluation Pipeline

QUACK includes a fully automated evaluation pipeline that reads game logs and computes metrics across three tiers. Tier 1 and Tier 2 run purely from log data; Tier 3 uses an LLM to extract and verify natural language claims.

### Tier 1 — Game-Level Metrics

Computed directly from engine events. No reconstruction required.

| Metric | Description |
|--------|-------------|
| Winner / win reason | `goose`, `duck`, or `timeout` with reason |
| Game duration | Total ticks |
| Task completion rate | Goose tasks completed / total assigned |
| Kill count & timing | Total kills, first kill tick, avg inter-kill interval |
| Meeting counts | Body reports vs emergency meetings |
| Ejection accuracy | Fraction of ejections that removed a Duck |
| Final survival | Alive counts by role |

### Tier 2 — Behavioral Metrics

Requires tick-by-tick game state reconstruction. The `GameReconstructor` replays all events to build a `GameTimeline` that tracks every player's room, transit state, alive status, and action at every tick.

| Metric | Description |
|--------|-------------|
| Goose voting accuracy | Fraction of goose votes targeting a Duck (excluding skips) |
| Goose skip rate | Fraction of goose votes that were skip/null |
| Report latency | Avg ticks between a body appearing in a goose's room and the report |
| Task efficiency | Fraction of goose free-roam ticks spent doing tasks or moving toward task rooms |
| Spatial coverage | Avg distinct rooms visited per goose / per duck |
| Post-kill displacement | Hop distance between kill room and killer's room 3 ticks later |
| Self-report rate | Fraction of kills where the duck reported its own body |
| Cooldown utilization | Fraction of kill opportunities (cooldown=0, goose in room) not taken |

### Tier 3 — Statement Verification

The most novel component. For each meeting discussion message:

1. **Claim extraction** — An LLM parses each statement into structured claims (`location`, `sighting`, `activity`, `accusation`, `defense`) with temporal references and normalized room names. Extraction results are cached on disk (`tier3_extraction_cache.jsonl`) for reproducibility.
2. **Ground-truth verification** — Each claim is checked against the `GameTimeline` and raw events:
   - **Location** — default semantics are *presence* (`any_time`: visited the room at least once in the window). `most_time` / `entire_time` apply only when the speaker uses explicit majority/continuity phrasing. Ordered multi-room routes can be verified as a subsequence.
   - **Sighting** — co-location while both parties are alive, with visibility check; killer-alibi and death-tick guards apply.
   - **Activity** — matched against task/movement/report events (including room-specific traveling claims).
   - **Accusation** — tracked on separate *outcome* (hit/miss) and *groundedness* axes; accusations do **not** enter truthfulness aggregates.
3. **Metric computation** — Aggregates verdicts into truthfulness, deception, hallucination, and detection rates. Full per-claim details live in `tier3_claims.jsonl` (the audit file is the source of truth for Tier 3 counts).

| Metric | Description |
|--------|-------------|
| Goose truthfulness | Fraction of goose verifiable claims (`location`/`sighting`/`activity`) with verdict `true` |
| Duck truthfulness | Fraction of duck verifiable claims with verdict `true` |
| Spatial hallucination rate | Fraction of goose **`location` + `sighting`** claims that are `false` or `wrong_room` (not activity) |
| Deception rate | Fraction of duck verifiable claims that are `false` or `wrong_room` |
| Deception sophistication | Among duck deceptive claims, fraction that are `near_miss` (partially grounded alibis) |
| Accusation accuracy | Fraction of accusations where the accused was actually a Duck |
| Unsupported accusation rate | Fraction of accusations with no observational grounding (separate from accuracy) |
| Lie detection rate | Fraction of meetings where a duck lied and was subsequently voted for |
| Per-player breakdown | Claim counts and verdicts per player |

**Note:** `duck_truthfulness + deception_rate + near_miss_rate ≈ 1` over verifiable claims — they are not simple complements because `near_miss` sits in the denominator but not in either numerator.

### Batch Evaluation

`evaluate_batch.py` recursively discovers `game.jsonl` files, groups results by experiment condition (`homogeneous/<model>` or `heterogeneous/<pair>`), and writes per-condition and overall aggregates (mean ± std):

```bash
# Tier 1 + 2 only
python scripts/evaluate_batch.py game_logs/

# Full pipeline including Tier 3 (uses api_key.txt if present)
python scripts/evaluate_batch.py game_logs/ --tier3
```

Results are saved as:
- **Per-game**: `evaluation.json` + (with `--tier3`) `tier3_claims.jsonl` and `tier3_extraction_cache.jsonl` alongside each `game.jsonl`
- **Per-condition**: `<search_root>/evaluation/{homogeneous_<model>|heterogeneous_<pair>}.json`
- **Overall**: `<search_root>/evaluation/summary.json`

Condition grouping is anchored on the `homogeneous/` or `heterogeneous/` segment in the log path, so it works whether you pass `game_logs/`, a category folder, or a single model directory.

### Programmatic Usage

```python
from quack.evaluation.evaluator import GameEvaluator

# Single game
evaluator = GameEvaluator()
result = evaluator.evaluate("game_logs/homogeneous/gpt5.5/20260520_230743_seed1/game.jsonl")
print(result.tier1.winner)           # "goose"
print(result.tier2.task_efficiency)  # 0.45

# With Tier 3 (writes tier3_claims.jsonl by default)
result = evaluator.evaluate(
    "game_logs/homogeneous/gpt5.5/20260520_230743_seed1/game.jsonl",
    run_tier3=True,
    llm_api_key="sk-...",
    llm_model="gpt-5.5",
)
print(result.tier3.goose_truthfulness)
print(result.tier3.spatial_hallucination_rate)
```

For batch evaluation over nested `game_logs/` trees, prefer `python scripts/evaluate_batch.py` over the `BatchEvaluator` class (which only scans flat `*.jsonl` in a single directory).

---

## Project Architecture

```
QUACK/
├── quack/
│   ├── engine/
│   │   ├── game_engine.py     # Main game loop, phase management, action dispatch
│   │   ├── game_state.py      # Core data structures (Player, GameState, Body, etc.)
│   │   └── event_bus.py       # Pub/sub event system
│   ├── map/
│   │   ├── game_map.py        # Discrete graph-based map with weighted corridors
│   │   └── pathfinding.py     # Dijkstra shortest path
│   ├── roles/
│   │   ├── base_role.py       # Abstract role with abilities
│   │   ├── goose.py           # Goose role (crew)
│   │   └── duck.py            # Duck role (impostor)
│   ├── systems/
│   │   ├── vision.py          # Same-room visibility, corridor vision
│   │   ├── task.py            # Task assignment and progress
│   │   ├── kill.py            # Kill mechanics and cooldowns
│   │   ├── meeting.py         # Body reports and emergency meetings
│   │   └── voting.py          # Vote tallying and ejection
│   ├── agents/
│   │   ├── base_agent.py      # Abstract agent interface
│   │   ├── vlm_agent.py       # VLM agent (OpenAI SDK, multi-model, streaming, memory)
│   │   ├── prompt_builder.py  # System prompts with role-specific strategy guides
│   │   └── memory.py          # Per-agent structured memory system
│   ├── evaluation/
│   │   ├── log_parser.py                    # Parse JSONL logs into structured event lists
│   │   ├── game_reconstructor.py            # Tick-by-tick state reconstruction (GameTimeline)
│   │   ├── tier1_game_metrics.py            # Game-level metrics (outcomes, tasks, kills, meetings)
│   │   ├── tier2_behavioral.py              # Behavioral metrics (spatial, voting, task efficiency)
│   │   ├── tier3_statement_verification.py  # LLM claim extraction + ground-truth verification
│   │   ├── evaluator.py                     # Orchestrator (GameEvaluator, BatchEvaluator)
│   │   └── report.py                        # Human-readable + JSON report generation
│   ├── rendering/
│   │   ├── map_renderer.py    # Global, local, god view + meeting frame renderer
│   │   ├── sprites.py         # Procedural pixel-art character sprites
│   │   ├── room_decor.py      # Themed pixel-art room decorations
│   │   └── colors.py          # Color palette and constants
│   └── utils/
│       ├── config.py          # YAML config loader
│       └── logger.py          # Structured JSONL game logger
├── configs/
│   ├── config.yaml            # Hydra main config (defaults + runtime overrides)
│   ├── game/
│   │   └── default.yaml       # Game rules (num_players, max_ticks, tasks, kill, etc.)
│   ├── map/
│   │   └── simple_ship.yaml   # 10-room ship map with weighted corridors
│   ├── model/
│   │   ├── gpt5.5.yaml            # GPT-5.5 (default)
│   │   ├── claude_opus4.7.yaml    # Claude Opus 4.7
│   │   └── gemini3.1pro.yaml      # Gemini 3.1 Pro Preview (streaming)
│   ├── experiment/
│   │   ├── homogeneous.yaml   # All players same model
│   │   └── heterogeneous.yaml # Different models for geese vs duck
│   └── maps/
│       └── simple_ship.yaml   # Legacy map path (kept for backward compat)
├── scripts/
│   ├── run_game.py            # Hydra CLI entry point (multi-model, auto video)
│   ├── replay_game.py         # Replay a single game log → renders/video
│   ├── replay_all.sh          # Batch re-render all game.jsonl under game_logs/
│   ├── evaluate_game.py       # Evaluate a single game log (all tiers)
│   ├── evaluate_batch.py      # Recursive batch evaluate with per-condition aggregation
│   ├── merge_batch_results.py # Merge multiple batch-eval JSON files
│   ├── validate_tier3_audit.py    # Tier 3 audit ↔ summary consistency checks
│   ├── reverify_tier3_from_audit.py  # Re-run verification from tier3_claims.jsonl
│   ├── normalize_map_paths.py # Fix absolute map paths in old game.jsonl logs
│   ├── batch_homogeneous.sh   # Batch runner for homogeneous experiments
│   ├── batch_heterogeneous.sh # Batch runner for cross-model experiments
│   ├── batch_full_experiment.sh # One-command full experiment suite
│   └── generate_videos.sh     # Re-encode video.mp4 from existing god-view frames
├── tests/
│   └── test_evaluation/       # Unit tests for the evaluation pipeline
├── game_logs/                 # Experiment output (structured by model/condition)
├── api_key.txt                # Greatrouter API key for GPT-5.5/Gemini (not committed)
├── claude_key.txt             # Anthropic API key for Claude Opus 4.7 (not committed)
├── pyproject.toml
└── README.md
```

---

## Links

- **Paper (arXiv):** [https://arxiv.org/abs/2605.27068](https://arxiv.org/abs/2605.27068)
- **Paper (Hugging Face):** [https://huggingface.co/papers/2605.27068](https://huggingface.co/papers/2605.27068)
- **Game logs (Hugging Face Dataset):** [https://huggingface.co/datasets/5a-academia-attractions/QUACK](https://huggingface.co/datasets/5a-academia-attractions/QUACK)

## Citation

If you use QUACK in your research, please cite our paper:

```bibtex
@misc{yuan2026quack,
      title={QUACK: Questioning, Understanding, and Auditing Communicated Knowledge in Multimodal Social Deduction Agents},
      author={Ye Yuan and Rui Song and Weien Li and Zeyu Li and Haochen Liu and Xiangyu Kong and Changjiang Han and Yonghan Yang and Zichen Zhao and Zixuan Dong and Fuyuan Lyu and Bowei He and Haolun Wu and Jikun Kang and Xue Liu},
      year={2026},
      eprint={2605.27068},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.27068},
}
```
