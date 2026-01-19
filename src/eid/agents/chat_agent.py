"""CAMEL ChatAgent wrapper for EID-Benchmark.

Provides a unified interface for interacting with LLM agents across different roles.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory

from eid.config import ModelConfig

logger = logging.getLogger(__name__)


class AgentWrapper:
    """Wrapper around CAMEL ChatAgent with role-specific configuration.

    Attributes:
        role_id: Identifier for this agent's role (e.g., 'doctor', 'patient')
        system_prompt: System message defining the agent's behavior
        config: Model configuration
    """

    def __init__(
        self,
        role_id: str,
        system_prompt: str,
        config: ModelConfig,
        message_window_size: int | None = None,
        summarize_threshold: int | None = None,
    ) -> None:
        """Initialize the agent wrapper.

        Args:
            role_id: Identifier for this agent's role
            system_prompt: System message defining agent behavior
            config: Model configuration
            message_window_size: Maximum messages in context window
            summarize_threshold: Context percentage that triggers summarization
        """
        self.role_id = role_id
        self.system_prompt = system_prompt
        self.config = config
        self._last_token_usage: dict[str, int] | None = None
        self._last_duration: float | None = None

        # Build CAMEL model configuration
        camel_config = config.to_camel_config()
        model_config_dict = camel_config.pop("model_config_dict", {})

        # Extract model parameters
        model_kwargs: dict[str, Any] = {
            "model_platform": camel_config["model_platform"],
            "model_type": camel_config["model_type"],
            "model_config_dict": {
                "temperature": camel_config.get("temperature", 0.0),
                "max_tokens": camel_config.get("max_tokens", 6000),
                **model_config_dict,
            },
        }

        if camel_config.get("api_key"):
            model_kwargs["api_key"] = camel_config["api_key"]
        if camel_config.get("url"):
            model_kwargs["url"] = camel_config["url"]
        if camel_config.get("timeout"):
            model_kwargs["timeout"] = camel_config["timeout"]
        if camel_config.get("max_retries"):
            model_kwargs["max_retries"] = camel_config["max_retries"]

        model_inst = ModelFactory.create(**model_kwargs)

        # Initialize ChatAgent
        agent_kwargs: dict[str, Any] = {
            "system_message": system_prompt,
            "model": model_inst,
        }
        if message_window_size is not None:
            agent_kwargs["message_window_size"] = message_window_size
        if summarize_threshold is not None:
            agent_kwargs["summarize_threshold"] = summarize_threshold

        self._agent = ChatAgent(**agent_kwargs)

    def step(self, instruction: str) -> str:
        """Send a message to the agent and get a response.

        Args:
            instruction: User message to send

        Returns:
            Agent's response content

        Raises:
            RuntimeError: If all retry attempts fail
        """
        user_message = BaseMessage.make_user_message(role_name="user", content=instruction)

        attempt = 0
        last_error: Exception | None = None
        max_retries = self.config.max_retries

        while attempt < max_retries:
            try:
                start_time = time.time()
                response = self._agent.step(user_message)
                self._last_duration = time.time() - start_time
                
                if not response.msgs:
                    raise ValueError("Agent returned no messages")

                response_msg = response.msg if hasattr(response, "msg") else response.msgs[0]
                return str(response_msg.content)

            except Exception as exc:
                attempt += 1
                last_error = exc
                logger.warning(
                    "Agent '%s' attempt %s/%s failed: %s",
                    self.role_id,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt >= max_retries:
                    break
                # Random backoff
                wait_time = random.uniform(0, 5)
                time.sleep(wait_time)

        raise RuntimeError(f"Agent '{self.role_id}' failed after {max_retries} retries") from last_error

    def get_last_duration(self) -> float | None:
        """Get duration of last step call."""
        return self._last_duration

    def record_message(self, role: str, content: str) -> None:
        """Record a message in the agent's memory.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        if role == "user":
            msg = BaseMessage.make_user_message(role_name="user", content=content)
        else:
            msg = BaseMessage.make_assistant_message(role_name="assistant", content=content)
        self._agent.record_message(msg)

    def get_history(self) -> list[dict[str, Any]]:
        """Get conversation history.

        Returns:
            List of message dictionaries
        """
        messages, token_count = self._agent.memory.get_context()
        self._last_token_usage = self._format_token_usage(token_count)

        history = []
        for message in messages:
            history.append({
                "role": message.get("role"),
                "content": message.get("content"),
            })
        return history

    def get_usage_stats(self) -> dict[str, int] | None:
        """Get token usage statistics.

        Returns:
            Token usage dictionary or None
        """
        if self._last_token_usage is not None:
            return self._last_token_usage

        _, token_count = self._agent.memory.get_context()
        self._last_token_usage = self._format_token_usage(token_count)
        return self._last_token_usage

    def reset(self) -> None:
        """Reset agent memory."""
        self._agent.reset()
        self._last_token_usage = None
        self._last_duration = None

    def _format_token_usage(self, token_count: Any) -> dict[str, int] | None:
        """Normalize token count to dictionary format."""
        if token_count is None:
            return None

        if isinstance(token_count, dict):
            normalized: dict[str, int] = {}
            for key, value in token_count.items():
                if value is None:
                    continue
                try:
                    normalized[key] = int(value)
                except (TypeError, ValueError):
                    continue
            return normalized or None

        try:
            return {"total_tokens": int(token_count)}
        except (TypeError, ValueError):
            return None


def create_agent(
    role_id: str,
    system_prompt: str,
    config: ModelConfig,
    message_window_size: int | None = None,
    summarize_threshold: int | None = None,
) -> AgentWrapper:
    """Factory function to create an agent wrapper.

    Args:
        role_id: Identifier for the agent's role
        system_prompt: System message defining behavior
        config: Model configuration
        message_window_size: Maximum messages in context
        summarize_threshold: Context percentage for summarization

    Returns:
        Configured AgentWrapper instance
    """
    return AgentWrapper(
        role_id=role_id,
        system_prompt=system_prompt,
        config=config,
        message_window_size=message_window_size,
        summarize_threshold=summarize_threshold,
    )
