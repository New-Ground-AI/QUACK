"""Entry point to configure and run a full Goose Goose Duck game.

Uses Hydra for configuration management.  All CLI overrides follow the
``key=value`` syntax, e.g. ``model=claude_opus4.7 seed=42``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quack.agents.base_agent import BaseAgent
from quack.engine.game_engine import GameEngine
from quack.utils.logger import GameLogger

logger = logging.getLogger(__name__)

PLAYER_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
    "Grace", "Hank", "Ivy", "Jack",
]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class RandomAgent(BaseAgent):
    """Simple random agent for testing the game loop without VLM calls."""

    async def choose_action(self, observation: dict[str, Any], phase: str) -> str:
        actions = observation.get("available_actions", ["wait()"])
        action = random.choice(actions)
        return action.split("#")[0].strip()

    async def speak(self, observation: dict[str, Any]) -> str:
        current_room = observation.get("current_room", "somewhere")
        visible = observation.get("visible_players", [])
        phrases = [
            f"I was in {current_room} doing my tasks.",
            "I didn't see anything suspicious.",
            "I think we should skip this vote.",
            "Has anyone seen anything?",
        ]
        if visible:
            target = random.choice(visible)
            phrases.append(f"I saw {target['name']} acting suspicious near {target['room']}.")
        return random.choice(phrases)

    async def vote(self, observation: dict[str, Any]) -> str | None:
        votable = observation.get("votable_players", [])
        if random.random() < 0.3 or not votable:
            return None
        return random.choice(votable)["id"]


def create_random_agents(num_players: int) -> dict[str, BaseAgent]:
    agents: dict[str, BaseAgent] = {}
    for i in range(num_players):
        pid = f"player_{i}"
        name = PLAYER_NAMES[i % len(PLAYER_NAMES)]
        agents[pid] = RandomAgent(player_id=pid, name=name)
    return agents


def _resolve_provider_key(
    model_cfg: DictConfig,
    openai_key: str,
    anthropic_key: str,
) -> str:
    """Pick the API key matching the model's provider.

    Falls back to ``openai_key`` for any unknown provider so we don't break
    third-party model configs that don't set the ``provider`` field.
    """
    provider = model_cfg.get("provider", "openai")
    if provider == "anthropic":
        return anthropic_key
    return openai_key


def create_agents_from_config(
    cfg: DictConfig,
    api_key: str,
    num_players: int,
    anthropic_api_key: str = "",
) -> dict[str, BaseAgent]:
    """Create VLM agents based on experiment config.

    Homogeneous: all agents use ``cfg.model``.
    Heterogeneous: all agents initially created with the goose model;
    duck agents are swapped after role assignment via ``reassign_duck_agents``.

    ``api_key`` is the OpenAI-compatible (greatrouter) key.
    ``anthropic_api_key`` is the official Anthropic key; used only when a
    model config declares ``provider: anthropic``.
    """
    from quack.agents.vlm_agent import VLMAgent

    provider = cfg.model.get("provider", "openai")
    resolved_key = _resolve_provider_key(cfg.model, api_key, anthropic_api_key)

    agents: dict[str, BaseAgent] = {}
    for i in range(num_players):
        pid = f"player_{i}"
        name = PLAYER_NAMES[i % len(PLAYER_NAMES)]
        agents[pid] = VLMAgent(
            player_id=pid,
            name=name,
            api_key=resolved_key,
            base_url=cfg.base_url,
            model=cfg.model.model_id,
            temperature=cfg.model.get("temperature", None),
            speak_chinese=cfg.speak_chinese,
            requires_stream=cfg.model.get("requires_stream", False),
            provider=provider,
            max_tokens=cfg.model.get("max_tokens", None),
            send_images=cfg.get("send_images", True),
        )
    return agents


def reassign_duck_agents(
    engine: GameEngine,
    agents: dict[str, BaseAgent],
    cfg: DictConfig,
    api_key: str,
    original_cwd: str,
    anthropic_api_key: str = "",
) -> None:
    """After role assignment, replace duck players' agents with the duck model.

    Only used in heterogeneous experiments.
    """
    if cfg.experiment.type != "heterogeneous":
        return

    duck_model_cfg = OmegaConf.load(
        Path(original_cwd) / "configs" / "model" / f"{cfg.experiment.duck_model}.yaml"
    )

    from quack.agents.vlm_agent import VLMAgent

    duck_provider = duck_model_cfg.get("provider", "openai")
    duck_key = _resolve_provider_key(duck_model_cfg, api_key, anthropic_api_key)

    for pid, player in engine.state.players.items():
        if player.team.value == "duck":
            old_agent = agents[pid]
            new_agent = VLMAgent(
                player_id=pid,
                name=old_agent.name,
                api_key=duck_key,
                base_url=cfg.base_url,
                model=duck_model_cfg.model_id,
                temperature=duck_model_cfg.get("temperature", None),
                speak_chinese=cfg.speak_chinese,
                requires_stream=duck_model_cfg.get("requires_stream", False),
                provider=duck_provider,
                max_tokens=duck_model_cfg.get("max_tokens", None),
                send_images=cfg.get("send_images", True),
            )
            agents[pid] = new_agent
            engine.agents[pid] = new_agent


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def build_output_dir(cfg: DictConfig, base_cwd: str) -> Path:
    """Build the output directory path from experiment config."""
    base = Path(base_cwd) / cfg.output_base
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"seed{cfg.seed}" if cfg.seed is not None else "random"
    run_name = f"{timestamp}_{seed_str}"

    if cfg.experiment.type == "homogeneous":
        return base / "homogeneous" / cfg.model.name / run_name
    elif cfg.experiment.type == "heterogeneous":
        duck_model_name = cfg.experiment.duck_model
        condition_name = f"geese_{cfg.model.name}_duck_{duck_model_name}"
        return base / "heterogeneous" / condition_name / run_name
    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment.type}")


def generate_video(frames_dir: Path, output_path: Path, fps: int = 1) -> None:
    """Stitch god-view frames into an MP4 video via ffmpeg."""
    if not any(frames_dir.glob("frame_*.png")):
        logger.warning("No frames found in %s — skipping video generation", frames_dir)
        return
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info("Video saved to %s", output_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found — install ffmpeg to enable automatic video generation")
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg failed: %s", exc.stderr.decode(errors="replace")[:500])


# ---------------------------------------------------------------------------
# Build game-engine compatible config dict from Hydra DictConfig
# ---------------------------------------------------------------------------

def _build_engine_config(cfg: DictConfig, map_config_path: str) -> dict[str, Any]:
    """Translate the Hydra config to the flat dict expected by GameEngine."""
    return {
        "game": {
            "num_players": cfg.game.num_players,
            "num_ducks": cfg.game.num_ducks,
            "max_ticks": cfg.game.max_ticks,
            "map": map_config_path,
        },
        "tasks": OmegaConf.to_container(cfg.game.tasks, resolve=True),
        "kill": OmegaConf.to_container(cfg.game.kill, resolve=True),
        "meeting": OmegaConf.to_container(cfg.game.meeting, resolve=True),
        "vision": OmegaConf.to_container(cfg.game.vision, resolve=True),
    }


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

async def run_game(
    cfg: DictConfig,
    api_key: str,
    output_dir: Path,
    original_cwd: str,
    anthropic_api_key: str = "",
) -> None:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    map_config_path = str(Path(original_cwd) / "configs" / "maps" / "simple_ship.yaml")
    engine_config = _build_engine_config(cfg, map_config_path)
    engine = GameEngine.from_config_dict(engine_config, map_config_path)

    game_logger = GameLogger(log_path=output_dir / "game.jsonl")
    engine.event_bus.subscribe_all(game_logger.handle_event)

    engine.enable_rendering()

    num_players = cfg.game.num_players
    # A run uses VLM agents if we have *any* applicable key: the OpenAI-style
    # key covers gpt5.5/gemini3.1pro, the Anthropic key covers claude_opus4.7.
    use_vlm = bool(api_key) or bool(anthropic_api_key)

    if use_vlm:
        img_mode = "image+text" if cfg.get("send_images", True) else "TEXT-ONLY (images disabled)"
        print(
            f"Creating {num_players} VLM agents (model: {cfg.model.display_name}, "
            f"input: {img_mode})..."
        )
        agents = create_agents_from_config(
            cfg, api_key, num_players, anthropic_api_key=anthropic_api_key,
        )
    else:
        agents = create_random_agents(num_players)

    engine.register_agents(agents)
    await engine.setup_game()

    if use_vlm and cfg.experiment.type == "heterogeneous":
        reassign_duck_agents(
            engine, agents, cfg, api_key, original_cwd,
            anthropic_api_key=anthropic_api_key,
        )
        for pid, player in engine.state.players.items():
            if player.team.value == "duck":
                role = engine.roles[pid]
                duck_ids = {
                    p.player_id
                    for p in engine.state.players.values()
                    if p.team.value == "duck"
                }
                teammates = [
                    engine.state.players[did].name
                    for did in duck_ids
                    if did != pid
                ]
                await agents[pid].on_game_start(
                    role_name=role.name,
                    team=player.team.value,
                    objective=role.objective,
                    total_geese=engine.state.alive_goose_count,
                    total_ducks=engine.state.alive_duck_count,
                    teammates=teammates,
                    all_players=[p.name for p in engine.state.players.values()],
                )

    # Save frozen config
    config_snapshot = OmegaConf.to_yaml(cfg)
    (output_dir / "config.yaml").write_text(config_snapshot)

    print(f"Game started with {num_players} players")
    print(
        f"Roles: { {pid: f'{p.name} ({p.role_name}/{p.team.value})' for pid, p in engine.state.players.items()} }"
    )
    print(f"Log: {game_logger.log_path}")
    if use_vlm:
        for pid, p in engine.state.players.items():
            if p.team.value == "duck":
                agent = agents[pid]
                if hasattr(agent, "_teammates"):
                    print(f"  Duck {p.name} knows teammates: {agent._teammates}")
    print("-" * 60)

    god_frame_counter = [0]
    god_view_dir = output_dir / "renders" / "god_view"

    def _next_god_frame() -> int:
        god_frame_counter[0] += 1
        return god_frame_counter[0]

    # ---- Frame 1: tick-0 spawn snapshot ----
    # The engine increments the tick at the START of each free-roam step,
    # so without this dedicated frame the very first thing in the video
    # would be tick 1 (after agents have already moved). Render the spawn
    # state explicitly so the viewer can see where everyone started.
    if cfg.god_view and engine.renderer:
        _save_god_view_frame(
            engine,
            _next_god_frame(),
            god_view_dir,
            phase_override="Spawn",
        )

    while engine.state.phase.value != "game_over":
        prev_phase = engine.state.phase

        if engine.state.phase.value == "free_roam":
            await engine._run_free_roam_tick()
            if cfg.god_view and engine.renderer:
                _save_god_view_frame(engine, _next_god_frame(), god_view_dir)

            if use_vlm:
                tick = engine.state.current_tick
                if tick % 5 == 0:
                    print(f"  Tick {tick}...")
                for msgs in engine.state.room_messages.values():
                    for msg in msgs:
                        if msg.get("tick") == tick:
                            room = msg.get("room", "?")
                            name = msg.get("name", "?")
                            text = msg.get("message", "")
                            print(f"  [CHAT][{room}] {name}: {text}")

        elif engine.state.phase.value == "discussion":
            if (cfg.god_view and engine.renderer
                    and engine.state.current_speaker_idx == 0
                    and engine.state.discussion_round == 0
                    and len(engine.state.discussion_messages) == 0):
                meeting_img = engine.renderer.render_meeting_called(
                    engine.state,
                    engine.state.meeting_reason or "",
                    engine.state.current_tick,
                )
                _save_meeting_frame(meeting_img, _next_god_frame(), god_view_dir)

            await engine._run_discussion()

            if cfg.god_view and engine.renderer and engine.state.discussion_messages:
                last_msg = engine.state.discussion_messages[-1]
                speech_img = engine.renderer.render_speech(
                    engine.state,
                    last_msg["player_id"],
                    last_msg["message"],
                    engine.state.discussion_messages,
                    engine.state.current_tick,
                )
                _save_meeting_frame(speech_img, _next_god_frame(), god_view_dir)

            if use_vlm and engine.state.discussion_messages:
                last_msg = engine.state.discussion_messages[-1]
                speaker_name = engine.state.players[last_msg["player_id"]].name
                print(f"  [{speaker_name}]: {last_msg['message']}")

        elif engine.state.phase.value == "voting":
            await engine._run_voting()

            if cfg.god_view and engine.renderer:
                ejected_id = None
                for p in engine.state.players.values():
                    if not p.is_alive and any(
                        t == p.player_id for t in engine.state.votes.values()
                    ):
                        ejected_id = p.player_id
                        break
                vote_img = engine.renderer.render_vote_result(
                    engine.state,
                    engine.state.votes,
                    ejected_id,
                    engine.state.current_tick,
                )
                _save_meeting_frame(vote_img, _next_god_frame(), god_view_dir)

        elif engine.state.phase.value == "ejection":
            engine._post_ejection()
            # The engine clears bodies and randomly re-spawns every alive
            # player. Without a dedicated frame here, the next free-roam
            # frame would make players look like they "teleport" relative
            # to the last meeting/vote frame. Render the respawn snapshot
            # explicitly so the random repositioning is visible.
            if (cfg.god_view and engine.renderer
                    and engine.state.phase.value == "free_roam"):
                _save_god_view_frame(
                    engine,
                    _next_god_frame(),
                    god_view_dir,
                    phase_override="Respawn (post-meeting)",
                )

        engine._check_win_conditions()

        if engine.state.phase != prev_phase:
            print(f"  Phase: {prev_phase.value} -> {engine.state.phase.value}")

    from quack.engine.event_bus import EventType, GameEvent
    engine.event_bus.emit(GameEvent(
        event_type=EventType.GAME_OVER,
        data={
            "winner": engine.state.winner.value if engine.state.winner else None,
            "reason": engine.state.win_reason,
        },
        tick=engine.state.current_tick,
    ))

    print("-" * 60)
    winner = engine.state.winner
    print(f"Game Over! Winner: {winner.value if winner else 'None'}")
    print(f"Reason: {engine.state.win_reason}")
    print(f"Ticks: {engine.state.current_tick}")
    print(f"Log saved to: {game_logger.log_path}")
    print(f"Events recorded: {len(game_logger.get_entries())}")

    if cfg.video and cfg.god_view and god_view_dir.exists():
        video_path = output_dir / "video.mp4"
        print(f"Generating video at {video_path} ...")
        generate_video(god_view_dir, video_path, fps=cfg.video_fps)


def _save_god_view_frame(
    engine: GameEngine,
    frame_num: int,
    renders_dir: Path,
    phase_override: str | None = None,
) -> None:
    renders_dir.mkdir(parents=True, exist_ok=True)
    god_img = engine.render_god_view(
        frame_idx=frame_num,
        phase_override=phase_override,
    )
    if god_img:
        god_img.save(renders_dir / f"frame_{frame_num:04d}.png")


def _save_meeting_frame(img: object, frame_num: int, renders_dir: Path) -> None:
    renders_dir.mkdir(parents=True, exist_ok=True)
    if img and hasattr(img, "save"):
        img.save(renders_dir / f"frame_{frame_num:04d}.png")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    original_cwd = hydra.utils.get_original_cwd()

    api_key = cfg.api_key or os.environ.get("QUACK_API_KEY", "")
    if not api_key:
        key_path = Path(original_cwd) / "api_key.txt"
        if key_path.exists():
            api_key = key_path.read_text().strip()
            print(f"Loaded API key from {key_path}")

    # Separate Anthropic key for Claude Opus 4.7 (provider: anthropic).
    # Claude on the greatrouter proxy was heavily rate-limited, so we now hit
    # api.anthropic.com directly using a dedicated key stored in
    # ``claude_key.txt`` (or in ``$ANTHROPIC_API_KEY``).
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_api_key:
        claude_key_path = Path(original_cwd) / "claude_key.txt"
        if claude_key_path.exists():
            anthropic_api_key = claude_key_path.read_text().strip()
            print(f"Loaded Anthropic API key from {claude_key_path}")

    if api_key or anthropic_api_key:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    output_dir = build_output_dir(cfg, original_cwd)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    asyncio.run(run_game(
        cfg, api_key, output_dir, original_cwd,
        anthropic_api_key=anthropic_api_key,
    ))


if __name__ == "__main__":
    main()
