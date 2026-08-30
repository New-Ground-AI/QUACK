"""Constructs prompts from observations and memory for VLM agents."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from quack.agents.memory import AgentMemory


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# System prompts with embedded strategy guide
# ---------------------------------------------------------------------------

_GOOSE_STRATEGY = """\
STRATEGY GUIDE (Goose / Innocent):

Your primary goal is to complete tasks and identify Ducks.

[During Free Roam]
- Prioritize completing your tasks efficiently — plan your route based on distance.
- Try to find another player to travel with ("buddy up"). Traveling in a pair makes \
you safer and gives you a witness who can vouch for you.
- Pay attention to who you see, where, and what they're doing. Were they doing tasks? \
Just passing through? Following you? This information is crucial for meetings.
- If you find a dead body, report it immediately with report().

[During Discussion — Early Speaker (first 2-3 speakers)]
- Your job is to provide information for later speakers to analyze.
- Clearly state: your route (which rooms you visited in order), who you encountered \
and where, what those people were doing, whether anyone was following you or ran away, \
whether you saw the victim and when.
- If you reported the body, state exactly where you found it.
- A "clean" early speech is just providing solid information — don't accuse without \
evidence yet.

[During Discussion — Late Speaker (last 2-3 speakers)]
- Listen carefully to everything said before you. Take notes mentally.
- Identify: who has confirmed alibis, who was near the crime scene, whose story \
contradicts others, who is being suspiciously vague.
- If you're the last speaker, you have the most information — try to synthesize and \
suggest who to vote for. Other undecided players will likely follow your lead.
- If you're not confident, say so honestly and suggest skipping the vote.

[Voting]
- Vote based on evidence and logical deduction, not emotion.
- Consider who had opportunity (near the body) and whose story has holes.
- If you have a buddy who can vouch for you, mention that to defend yourself.
- Skip if there's genuinely no strong suspect — a wrong vote helps the Ducks.
"""

_DUCK_STRATEGY = """\
STRATEGY GUIDE (Duck / Impostor):

Your primary goal is to eliminate Geese until Ducks have voting majority.

[During Free Roam]
- After killing someone, IMMEDIATELY move to a different room. Never stay in the room \
where you killed. The further away you are when the body is found, the safer you are.
- At the start, look for isolated players to kill. Geese tend to buddy up quickly, \
so early kills on lone players are easiest.
- If you have a Duck teammate, find them and travel together. When you encounter a \
similar-sized group of Geese, strike decisively with your teammate (double kill).
- Be decisive when killing — hesitation gets you caught. If you see a good opportunity, \
take it.
- You don't have real tasks. You can use do_task() to fake doing tasks — this builds \
your alibi, but don't waste too much time on it. Prioritize hunting targets.
- Move through rooms to build a plausible route you can describe in meetings.

[Reporting Bodies — Strategic Decision]
- Generally, do NOT report bodies, especially ones you killed. Leave the area and \
let a Goose find it — reporters are always suspects.
- EXCEPTION: If you walk into a room containing a body AND a Goose is right there \
with you, it may be SMARTER to report it. The Goose witness can vouch that you \
reported immediately upon seeing it, making you look innocent. Failing to report \
when a Goose is watching would look extremely suspicious.
- Think carefully: who is watching? If no one sees you near the body, leave silently. \
If someone is with you, reporting can clear your name.

[During Discussion]
- Build your cover story: describe your route, mention rooms you visited, claim you \
were doing tasks. Mix real information with strategically omitted details.
- NEVER sell out your teammate. Always protect them. Simple ways to defend a teammate:
  * "I walked alone with X and they didn't kill me — I can half-vouch for them."
  * "X and I were together the whole time, they had no chance to act."
  * "X told me their role earlier, I believe them this round."
- If your teammate is under heavy suspicion and you can't save them directly, deflect \
subtly: "I haven't seen X this round, no info there. But Y was following me and \
asking my role — that felt suspicious."
- If two Geese are arguing with each other, fan the flames gently. Support one side \
to deepen the conflict, but don't be too obvious about it.
- When YOU are the suspect, don't go quiet — speak up. Describe your route, your \
tasks, act like you're helping the team analyze. Silence when accused = death.
- Important info to withhold or be vague about: body location (if you know), your \
teammate's route, victim's identity if you know it.

[Voting]
- Vote strategically. Push votes toward Geese.
- If the group is split, vote with the larger faction to avoid suspicion.
- Coordinate with your teammate subtly — if you can both vote the same Goose, do it.
- Only skip if skipping serves you better than a wrong vote.

CRITICAL RULES:
1. Never reveal that you are a Duck. Never break character.
2. Never stay near a body you created. Move away immediately after killing.
"""


def build_system_prompt(
    player_name: str,
    role_name: str,
    team: str,
    objective: str,
    *,
    total_geese: int = 0,
    total_ducks: int = 0,
    teammates: list[str] | None = None,
    all_players: list[str] | None = None,
    speak_chinese: bool = False,
    send_images: bool = True,
) -> str:
    """Build the full system prompt with role info, rules, and strategy.

    ``send_images`` controls the "YOUR VISION" section: when True (default)
    the prompt tells the agent it receives a global-map image and a local-view
    image each tick. When False (text-only ablation) the prompt instead
    explains that all spatial information is provided as structured text, so
    the model is not told to look for images that were never attached.
    """

    role_display = "Goose (Innocent)" if team == "goose" else "Duck (Impostor)"
    strategy = _GOOSE_STRATEGY if team == "goose" else _DUCK_STRATEGY

    team_info = f"Team composition: {total_geese} Geese, {total_ducks} Ducks."
    if all_players:
        team_info += f"\nAll players in this game: {', '.join(all_players)}."
    if teammates and team == "duck":
        team_info += f"\nYour Duck teammates: {', '.join(teammates)}. Protect them."

    if send_images:
        vision_info = (
            "YOUR VISION: You receive two images each tick:\n"
            "1. A global map showing the ship's room layout and your task "
            "locations (marked with your color). You CANNOT see other players "
            "on this map.\n"
            "2. A local view showing ONLY your current room and its immediate "
            "surroundings — players and bodies you can actually see right "
            "now.\n\n"
        )
    else:
        vision_info = (
            "YOUR VISION: All spatial information is provided to you as "
            "structured text (no images). Each tick you are told your current "
            "room, the adjacent rooms and travel times, your task locations, "
            "and the players and bodies you can actually see right now. Use "
            "this text description to reason about the ship layout and who is "
            "around you.\n\n"
        )

    return (
        f"You are {player_name}, playing Goose Goose Duck — a social deduction game "
        f"similar to Among Us / Werewolf.\n"
        f"Your role: {role_display}\n"
        f"Your objective: {objective}\n"
        f"{team_info}\n\n"
        f"GAME RULES:\n"
        f"- The game alternates between Free Roam and Meetings.\n"
        f"- During Free Roam, players move between rooms on a ship map. Rooms are "
        f"connected by corridors with varying travel times (measured in ticks).\n"
        f"- Geese have tasks assigned to specific rooms. Go to the room and use "
        f"do_task() to work on them. Stay in the room until the task completes.\n"
        f"- Ducks can kill Geese when they're in the same room (limited by cooldown).\n"
        f"- You can only see players in rooms within your vision range.\n"
        f"- When a body is found (report()) or emergency bell is rung (call_meeting()), "
        f"all players enter a Meeting.\n"
        f"- During a Meeting, players speak one by one in order, then all vote "
        f"simultaneously. The player with the most votes is ejected.\n"
        f"- After a meeting, all living players are randomly respawned to new rooms "
        f"and all bodies are cleared.\n"
        f"- Geese win by completing all tasks OR voting out all Ducks.\n"
        f"- Ducks win when they reach voting majority (Ducks ≥ Geese).\n\n"
        f"{vision_info}"
        f"{strategy}\n"
        f"RESPONSE FORMAT:\n"
        f"- For actions: respond with EXACTLY one action from the available list "
        f"(e.g. 'move(medbay)'). You may optionally append a free-roam chat using "
        f"' | say(your message)' to speak to players in your CURRENT ROOM only. "
        f"Examples: 'move(medbay)', 'do_task()', \"\n"
        f"  or 'move(medbay) | say(I saw a body in storage)'.\n"
        f"- For discussion: respond with natural language as your character would "
        f"speak in a meeting. Stay in character.\n"
        f"- For voting: respond with EXACTLY a player name to vote for, or 'skip' "
        f"to abstain. Just the name or 'skip', nothing else.\n"
        + (
            "\nLANGUAGE REQUIREMENT: During discussion, you MUST speak in Simplified "
            "Chinese (简体中文), NOT Traditional Chinese (繁體中文). All your meeting "
            "speeches must be in 简体中文. Actions and votes should still use the "
            "English format (e.g. 'move(medbay)', player names).\n"
            if speak_chinese else ""
        )
    )


# ---------------------------------------------------------------------------
# Action prompt (Free Roam)
# ---------------------------------------------------------------------------

def build_action_prompt(
    observation: dict[str, Any],
    memory: AgentMemory | None = None,
) -> str:
    lines = ["=== CURRENT SITUATION ==="]

    if observation.get("in_transit"):
        lines.append(f"You are currently traveling to {observation.get('moving_to', '?')}. "
                      f"Ticks remaining: {observation.get('move_ticks_remaining', '?')}")
    else:
        lines.append(f"You are in: {observation['current_room']}")

    adj = observation.get("adjacent_rooms_detail", [])
    if adj:
        adj_parts = [f"{r['room']} ({r['travel_ticks']} ticks)" for r in adj]
        lines.append(f"Adjacent rooms: {', '.join(adj_parts)}")
    else:
        adj_simple = observation.get("adjacent_rooms", [])
        if adj_simple:
            lines.append(f"Adjacent rooms: {', '.join(adj_simple)}")

    visible = observation.get("visible_players", [])
    if visible:
        player_strs: list[str] = []
        for p in visible:
            if p.get("in_transit"):
                co_dir = p.get("co_direction")
                if co_dir == "same":
                    player_strs.append(
                        f"{p['name']} (corridor, going SAME way as you)"
                    )
                elif co_dir == "opposite":
                    player_strs.append(
                        f"{p['name']} (corridor, going OPPOSITE way from you)"
                    )
                else:
                    player_strs.append(f"{p['name']} (in corridor)")
            else:
                player_strs.append(f"{p['name']} (in {p['room']})")
        lines.append(f"Visible players: {', '.join(player_strs)}")
    else:
        lines.append("Visible players: none — you are alone")

    transit_obs = observation.get("transit_observations", {}) or {}
    departures = transit_obs.get("departures", [])
    arrivals = transit_obs.get("arrivals", [])
    if departures or arrivals:
        lines.append("")
        lines.append("=== MOVEMENT AROUND YOU (this tick) ===")
        for ev in departures:
            mt = " (multi-tick)" if ev.get("multi_tick") else ""
            lines.append(f"  {ev['name']} LEFT toward {ev['to_room']}{mt}")
        for ev in arrivals:
            lines.append(f"  {ev['name']} ARRIVED from {ev['from_room']}")

    bodies = observation.get("visible_bodies", [])
    if bodies:
        body_strs = [f"{b['name']} (in {b['room']})" for b in bodies]
        lines.append(f"⚠ BODIES FOUND: {', '.join(body_strs)}")

    room_chat = observation.get("room_chat", [])
    if room_chat:
        lines.append("")
        lines.append("=== RECENT CHAT IN THIS ROOM (free roam) ===")
        for msg in room_chat[-5:]:
            speaker = msg.get("name", "?")
            text = msg.get("message", "")
            lines.append(f"  {speaker}: \"{text}\"")

    lines.append("")
    lines.append("=== YOUR TASKS ===")
    tasks = observation.get("tasks", [])
    if tasks:
        for t in tasks:
            if t["done"]:
                status = "COMPLETED"
            else:
                dist = t.get("distance_ticks", "?")
                status = f"progress {t['progress']}, ~{dist} ticks away"
            lines.append(f"  - {t['name']} @ {t['room']}: {status}")
    else:
        lines.append("  (no tasks assigned)")

    current_task = observation.get("current_task_here")
    if current_task:
        lines.append(f"  >> Active task in this room: {current_task['name']} "
                      f"({current_task['progress']})")

    cd = observation.get("kill_cooldown")
    if cd is not None:
        lines.append(f"\nKill cooldown: {cd} ticks remaining")

    if memory:
        lines.append("")
        lines.append("=== YOUR RECENT HISTORY ===")
        lines.append(memory.build_movement_summary(last_n=10))

    lines.append("")
    lines.append("=== AVAILABLE ACTIONS ===")
    for i, action in enumerate(observation.get("available_actions", []), 1):
        lines.append(f"  {i}. {action}")

    lines.append("")
    lines.append(
        "Choose ONE action. Respond with just the action (e.g. 'move(medbay)'). "
        "If you also want to speak to players in your CURRENT ROOM during Free Roam, "
        "append \" | say(your message)\". Example: "
        "'move(medbay) | say(I saw a body in storage)'."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discussion prompt
# ---------------------------------------------------------------------------

def build_discussion_prompt(
    observation: dict[str, Any],
    memory: AgentMemory | None = None,
) -> str:
    lines = [f"=== MEETING CALLED ==="]
    lines.append(f"Reason: {observation.get('meeting_reason', 'Unknown')}")

    dead = observation.get("dead_players", [])
    if dead:
        lines.append(f"Currently dead: {', '.join(dead)}")

    if observation.get("you_are_reporter") and observation.get("body_info"):
        lines.append("")
        lines.append("=== YOU ARE THE REPORTER (you called this meeting) ===")
        lines.append("The actual body locations (truth — you may choose to lie):")
        for info in observation["body_info"]:
            room = info.get("room", "?")
            victims = ", ".join(info.get("victims", []))
            lines.append(f"  - {room}: {victims}")
        lines.append("Describe what you found. Include the body location in your speech.")
        lines.append("(If you are a Duck, you may lie about the room or other details.)")

    lines.append("")
    history = observation.get("discussion_history", [])
    if history:
        lines.append("=== SPEECHES SO FAR ===")
        for msg in history:
            lines.append(f"  {msg['name']}: \"{msg['message']}\"")
    else:
        lines.append("No one has spoken yet. You are the first speaker.")

    speaker_order = observation.get("speaker_order", [])
    my_position = observation.get("my_position", -1)
    total_speakers = len(speaker_order) if speaker_order else 0
    if total_speakers > 0 and my_position >= 0:
        position_desc = "early" if my_position < total_speakers // 3 else (
            "middle" if my_position < 2 * total_speakers // 3 else "late"
        )
        lines.append(f"\nYou are speaker {my_position + 1} of {total_speakers} ({position_desc} position).")

    if memory:
        lines.append("")
        lines.append("=== YOUR OBSERVATIONS THIS ROUND ===")
        lines.append(f"Your route: {memory.get_route_description()}")
        lines.append(f"Player encounters:\n{memory.build_encounter_summary()}")
        witness_summary = memory.build_witness_summary()
        if witness_summary:
            lines.append(f"Witnessed movements:\n{witness_summary}")
        if memory.meeting_history:
            lines.append(f"\nPrevious meetings:\n{memory.build_meeting_summary()}")

    lines.append("")
    lines.append("It's your turn to speak. Share your observations and analysis. "
                  "Stay in character — speak as your player would in a meeting.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vote prompt
# ---------------------------------------------------------------------------

def build_vote_prompt(
    observation: dict[str, Any],
    memory: AgentMemory | None = None,
) -> str:
    lines = ["=== VOTING PHASE ==="]
    lines.append("All speeches are done. Time to vote.\n")

    history = observation.get("discussion_history", [])
    if history:
        lines.append("=== FULL DISCUSSION ===")
        for msg in history:
            lines.append(f"  {msg['name']}: \"{msg['message']}\"")
        lines.append("")

    dead = observation.get("dead_players", [])
    if dead:
        lines.append(f"Dead players: {', '.join(dead)}")

    if memory:
        witness_summary = memory.build_witness_summary()
        if witness_summary:
            lines.append(f"\nYour witnessed movements this round:\n{witness_summary}")
        if memory.meeting_history:
            lines.append(f"\nPast meetings:\n{memory.build_meeting_summary()}")

    lines.append("\n=== VOTABLE PLAYERS ===")
    for p in observation.get("votable_players", []):
        lines.append(f"  - {p['name']}")

    lines.append("\nVote for ONE player by typing their name, or say 'skip' to abstain.")
    lines.append("Respond with just the name or 'skip'.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message formatting for VLM API
# ---------------------------------------------------------------------------

def build_vlm_messages(
    system_prompt: str,
    user_text: str,
    images: list[Image.Image] | None = None,
) -> list[dict[str, Any]]:
    """Build the message list for an OpenAI-compatible VLM API call."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    content_parts: list[dict[str, Any]] = []

    if images:
        for img in images:
            b64 = image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

    content_parts.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": content_parts})

    return messages
