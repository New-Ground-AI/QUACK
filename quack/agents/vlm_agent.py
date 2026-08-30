"""VLM-powered agent using OpenAI SDK for Goose Goose Duck."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Any

from PIL import Image

from quack.agents.action_format import combine_action_and_say
from quack.agents.base_agent import BaseAgent
from quack.agents.memory import AgentMemory
from quack.agents.prompt_builder import (build_action_prompt,
                                         build_discussion_prompt,
                                         build_system_prompt,
                                         build_vlm_messages,
                                         build_vote_prompt)

logger = logging.getLogger(__name__)

logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# Substrings that indicate a transient failure worth retrying. Matched against
# the lowercased exception text so we catch errors that are wrapped by the
# greatrouter proxy (e.g. "upstreamException - {\"code\":\"quota_exceeded\"}")
# even when the SDK does not surface a typed exception.
_RETRYABLE_ERROR_SUBSTRINGS: tuple[str, ...] = (
    # Rate limiting / throttling
    "rate", "429", "retry-after", "too many requests",
    # Upstream proxy hiccups observed on greatrouter
    "quota_exceeded", "upstreamexception", "upstream exception",
    "upstream_exception", "upstream error", "upstream timeout",
    # 5xx server errors
    "500", "502", "503", "504",
    "internal server error", "bad gateway", "service unavailable",
    "gateway timeout",
    # Network / connection / streaming timeouts
    "timeout", "timed out", "read error", "connection",
    "broken pipe", "reset by peer", "remote end closed",
    "remoteprotocolerror",
)


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a transient error worth retrying.

    Uses typed-exception matching where the SDKs are available, and falls
    back to substring matching on the message so we also catch errors that
    have been wrapped by the upstream proxy (e.g. ``APIStatusError`` carrying
    a ``quota_exceeded`` payload).
    """
    try:
        import openai  # type: ignore
        if isinstance(
            exc,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        ):
            return True
    except Exception:
        pass

    try:
        import anthropic  # type: ignore
        if isinstance(
            exc,
            (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        ):
            return True
    except Exception:
        pass

    try:
        import httpx  # type: ignore
        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        ):
            return True
    except Exception:
        pass

    msg = str(exc).lower()
    return any(k in msg for k in _RETRYABLE_ERROR_SUBSTRINGS)


class VLMAgent(BaseAgent):
    """Agent driven by a Vision-Language Model via OpenAI-compatible API.

    Maintains per-game memory for strategic context.
    A class-level rate limiter ensures minimum spacing between API calls
    across all VLMAgent instances to avoid 429 errors.
    """

    _last_call_time: float = 0.0
    _min_call_interval: float = 1.0

    def __init__(
        self,
        player_id: str,
        name: str,
        api_key: str,
        base_url: str = "https://endpoint.greatrouter.com",
        model: str = "gpt-5.5",
        temperature: float | None = None,
        speak_chinese: bool = False,
        requires_stream: bool = False,
        provider: str = "openai",
        max_tokens: int | None = None,
        send_images: bool = True,
    ):
        super().__init__(player_id, name)
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.speak_chinese = speak_chinese
        self.requires_stream = requires_stream
        # Text-only ablation switch. When False, the rendered global map and
        # local view images are NOT attached to VLM requests; the agent then
        # relies solely on the structured text observation. The images are
        # still received via ``set_images`` (the engine renders them for the
        # god-view video regardless) but ``_collect_images`` drops them.
        self.send_images = send_images
        # Anthropic's Messages API requires ``max_tokens`` on every call.
        # OpenAI-compatible providers (greatrouter) ignore it for non-streaming
        # frontier models, so we only forward it when the model config sets it.
        self.max_tokens = max_tokens

        self._system_prompt = ""
        self.memory = AgentMemory(name)

        self._global_map_image: Image.Image | None = None
        self._local_view_image: Image.Image | None = None

        self._teammates: list[str] = []
        self._role_name = ""
        self._team = ""

        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                # Hit api.anthropic.com directly — the SDK uses that by
                # default, no base_url override needed.
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    max_retries=3,
                    timeout=60.0,
                )
            else:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=3,
                    timeout=60.0,
                )
        return self._client

    async def on_game_start(
        self,
        role_name: str,
        team: str,
        objective: str,
        *,
        total_geese: int = 0,
        total_ducks: int = 0,
        teammates: list[str] | None = None,
        all_players: list[str] | None = None,
    ) -> None:
        self._role_name = role_name
        self._team = team
        self._teammates = teammates or []
        self._system_prompt = build_system_prompt(
            self.name,
            role_name,
            team,
            objective,
            total_geese=total_geese,
            total_ducks=total_ducks,
            teammates=teammates,
            all_players=all_players,
            speak_chinese=self.speak_chinese,
            send_images=self.send_images,
        )
        self.memory = AgentMemory(self.name)

    def set_images(
        self,
        global_map: Image.Image | None = None,
        local_view: Image.Image | None = None,
    ) -> None:
        self._global_map_image = global_map
        self._local_view_image = local_view

    async def choose_action(self, observation: dict[str, Any], phase: str) -> str:
        self._record_observation(observation)

        user_text = build_action_prompt(observation, self.memory)
        images = self._collect_images()
        messages = build_vlm_messages(self._system_prompt, user_text, images)

        response = await self._call_vlm(messages)
        action = self._parse_action(response, observation.get("available_actions", []))
        result = combine_action_and_say(action, response)

        self.memory.tick_history[-1].action = result
        logger.info("[%s] action=%s (raw: %s)", self.name, result, response[:100])
        return result

    async def speak(self, observation: dict[str, Any]) -> str:
        user_text = build_discussion_prompt(observation, self.memory)
        images = self._collect_images()
        messages = build_vlm_messages(self._system_prompt, user_text, images)

        response = await self._call_vlm(messages)
        speech = response.strip()

        self.memory.record_my_speech(speech)
        logger.info("[%s] speech: %s", self.name, speech[:200])
        return speech

    async def vote(self, observation: dict[str, Any]) -> str | None:
        user_text = build_vote_prompt(observation, self.memory)
        images = self._collect_images()
        messages = build_vlm_messages(self._system_prompt, user_text, images)

        response = await self._call_vlm(messages)
        result = self._parse_vote(response, observation.get("votable_players", []))
        logger.info("[%s] vote=%s (raw: %s)", self.name, result, response[:100])
        return result

    def _record_observation(self, observation: dict[str, Any]) -> None:
        """Record tick observation into memory."""
        players_seen = [p["name"] for p in observation.get("visible_players", [])]
        bodies_seen = [b["name"] for b in observation.get("visible_bodies", [])]
        room_chat = [
            f"{msg.get('name', '?')}: {msg.get('message', '')}"
            for msg in observation.get("room_chat", [])
        ]
        transit_obs = observation.get("transit_observations", {}) or {}
        departures = list(transit_obs.get("departures", []))
        arrivals = list(transit_obs.get("arrivals", []))
        self.memory.record_tick(
            tick=observation.get("tick", 0),
            room=observation.get("current_room", "?"),
            action="",
            players_seen=players_seen,
            bodies_seen=bodies_seen,
            chats_heard=room_chat,
            in_transit=observation.get("in_transit", False),
            moving_to=observation.get("moving_to", ""),
            departures=departures,
            arrivals=arrivals,
        )

    def _collect_images(self) -> list[Image.Image]:
        # Text-only ablation: skip attaching any images to the VLM request.
        if not self.send_images:
            return []
        imgs: list[Image.Image] = []
        if self._global_map_image:
            imgs.append(self._global_map_image)
        if self._local_view_image:
            imgs.append(self._local_view_image)
        return imgs

    async def _call_vlm(self, messages: list[dict[str, Any]]) -> str:
        """Call the VLM API with global rate limiting and retry.

        Retries on transient failures (rate limits, timeouts, upstream
        ``quota_exceeded`` / ``upstreamException`` hiccups, 5xx, and network
        errors) with capped exponential backoff and jitter. Permanent client
        errors (400 ``BadRequestError``, 401, 403, ...) are not retried.
        """
        now = time.monotonic()
        wait = VLMAgent._min_call_interval - (now - VLMAgent._last_call_time)
        if wait > 0:
            await asyncio.sleep(wait)
        VLMAgent._last_call_time = time.monotonic()

        client = self._get_client()
        oai_messages = self._convert_messages(messages)

        max_retries = 4
        for attempt in range(max_retries):
            try:
                if self.requires_stream:
                    return await self._call_vlm_stream(client, oai_messages)
                else:
                    return await self._call_vlm_sync(client, oai_messages)
            except Exception as e:
                retryable = _is_retryable_error(e)
                if retryable and attempt < max_retries - 1:
                    # Capped exponential backoff with jitter so concurrent
                    # agents don't all retry on the same boundary.
                    backoff = min(30.0, 2.0 ** (attempt + 1)) + random.uniform(0, 0.75)
                    logger.warning(
                        "[%s] Transient VLM error %s (attempt %d/%d), "
                        "backing off %.1fs: %s",
                        self.name, type(e).__name__,
                        attempt + 1, max_retries, backoff,
                        str(e)[:240],
                    )
                    await asyncio.sleep(backoff)
                    VLMAgent._last_call_time = time.monotonic()
                    continue
                # Permanent error, or retries exhausted: log the full
                # traceback once and give up. Caller falls back to wait()
                # for actions / empty string for speech & votes.
                logger.exception(
                    "[%s] VLM API call failed (attempt %d/%d, retryable=%s)",
                    self.name, attempt + 1, max_retries, retryable,
                )
                return ""
        return ""

    def _build_create_kwargs(self, oai_messages: list[dict]) -> dict[str, Any]:
        """Build kwargs for client.chat.completions.create.

        Omits ``temperature`` when it's None. Frontier models on greatrouter
        (gpt-5.5, claude-opus-4-7, gemini-3.1-pro-preview) only accept the
        default temperature, so passing any value raises a 400.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    async def _call_vlm_sync(self, client: Any, oai_messages: list[dict]) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic_sync(client, oai_messages)
        response = client.chat.completions.create(
            **self._build_create_kwargs(oai_messages),
        )
        return response.choices[0].message.content or ""

    async def _call_vlm_stream(self, client: Any, oai_messages: list[dict]) -> str:
        if self.provider == "anthropic":
            # No Claude model currently requires streaming; if a future one
            # does, accumulate text deltas from client.messages.stream.
            return self._call_anthropic_sync(client, oai_messages)
        response = client.chat.completions.create(
            **self._build_create_kwargs(oai_messages),
            stream=True,
        )
        result = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content
        return result

    # ------------------------------------------------------------------
    # Anthropic-specific call path
    # ------------------------------------------------------------------

    def _call_anthropic_sync(self, client: Any, oai_messages: list[dict]) -> str:
        """Call the Anthropic Messages API.

        Translates the OpenAI-style messages used everywhere else in the
        codebase to Anthropic's format: pull the ``system`` message out into
        a top-level parameter, and convert each ``image_url`` content part
        into an ``image`` block with a ``base64`` source.
        """
        system_prompt, anth_messages = self._to_anthropic_messages(oai_messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            # Anthropic requires max_tokens. Fall back to a reasonable
            # default if the model config didn't set one.
            "max_tokens": self.max_tokens or 4096,
            "messages": anth_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        message = client.messages.create(**kwargs)
        if not getattr(message, "content", None):
            return ""
        # The response is a list of content blocks (text, tool_use, ...).
        # Concatenate all text blocks; tool_use blocks are not expected here.
        return "".join(
            getattr(block, "text", "")
            for block in message.content
            if getattr(block, "type", "") == "text"
        )

    @staticmethod
    def _to_anthropic_messages(
        oai_messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Translate OpenAI-format messages to Anthropic's Messages API format.

        Returns ``(system_prompt, anthropic_messages)`` where:

        * the ``system`` role is removed from the message list and returned
          separately (Anthropic uses a top-level ``system=`` param);
        * each ``image_url`` content part with a ``data:...;base64,...`` URL
          is rewritten as an ``image`` block whose ``source`` is the raw
          base64 payload and the matching ``media_type``;
        * text parts pass through unchanged.
        """
        system_chunks: list[str] = []
        out: list[dict[str, Any]] = []
        for msg in oai_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_chunks.append(content)
                elif isinstance(content, list):
                    system_chunks.extend(
                        part.get("text", "")
                        for part in content
                        if part.get("type") == "text"
                    )
                continue

            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            blocks: list[dict[str, Any]] = []
            for part in content or []:
                ptype = part.get("type")
                if ptype == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    media_type = "image/png"
                    data = url
                    if url.startswith("data:") and "," in url:
                        header, data = url.split(",", 1)
                        # header looks like "data:image/png;base64"
                        try:
                            media_type = header.split(":", 1)[1].split(";", 1)[0]
                        except Exception:
                            media_type = "image/png"
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
                elif ptype == "image":
                    # Already in Anthropic format — pass through.
                    blocks.append(part)
            out.append({"role": role, "content": blocks})

        system_prompt = "\n\n".join(s for s in system_chunks if s)
        return system_prompt, out

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message format to OpenAI SDK format."""
        converted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            elif isinstance(content, list):
                converted.append({"role": role, "content": content})
            else:
                converted.append(msg)
        return converted

    def _parse_action(self, response: str, available_actions: list[str]) -> str:
        """Extract a valid action from VLM response, with intelligent fallback."""
        response_clean = response.strip()

        for action in available_actions:
            action_clean = action.split("#")[0].strip()
            if action_clean in response_clean:
                return action_clean

        response_lower = response_clean.lower()
        for action in available_actions:
            action_clean = action.split("#")[0].strip()
            if action_clean.lower() in response_lower:
                return action_clean

        match = re.search(
            r'(move|do_task|kill|report|call_meeting|wait)\([^)]*\)',
            response_clean,
            re.IGNORECASE,
        )
        if match:
            extracted = match.group(0)
            for action in available_actions:
                action_clean = action.split("#")[0].strip()
                if action_clean.lower() == extracted.lower():
                    return action_clean

        for action in available_actions:
            action_base = action.split("(")[0].strip().lower()
            if action_base in response_lower and action_base != "wait":
                return action.split("#")[0].strip()

        return "wait()"

    def _parse_vote(self, response: str, votable_players: list[dict[str, Any]]) -> str | None:
        response_lower = response.strip().lower()

        if "skip" in response_lower or "abstain" in response_lower:
            return None

        for p in votable_players:
            if p["name"].lower() in response_lower:
                return p["id"]

        for p in votable_players:
            if p["id"].lower() in response_lower:
                return p["id"]

        return None
