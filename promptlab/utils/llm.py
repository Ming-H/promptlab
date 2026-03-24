"""
LLM client abstraction for PromptLab.

Provides a unified interface for calling different LLM providers.
"""

import os
from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for the given prompt."""
        pass


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize with predefined responses."""
        self.responses = responses or ["Default mock response"]
        self.call_count = 0

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Return predefined response in rotation."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        """Initialize the OpenAI client."""
        try:
            import openai
            self.openai = openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )
        self.model = model

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a completion using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
    ):
        """Initialize the Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic package is required. Install with: pip install anthropic"
            )
        self.model = model

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Generate a completion using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            text = response.content[0].text
            return text if isinstance(text, str) else ""
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e


def create_llm_client(
    provider: str = "mock",
    model: str | None = None,
    **kwargs: Any,
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: The LLM provider ("mock", "openai", "anthropic")
        model: The model to use
        **kwargs: Additional provider-specific arguments

    Returns:
        An LLMClient instance
    """
    if provider == "mock":
        return MockLLMClient()
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-haiku-20240307", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
