"""
LLM factory - creates provider-specific LLM instances.
"""

import os

from .config import (
    LLMProvider,
    OPENAI_AVAILABLE,
    ANTHROPIC_AVAILABLE,
    GOOGLE_AVAILABLE,
)


class LLMFactory:
    """Creates LLM instances based on provider configuration."""

    @staticmethod
    def create_llm(
        provider: LLMProvider,
        model: str = None,
        api_key: str = None,
        temperature: float = 0.1  # Lower temperature for medical accuracy
    ):
        """Create LLM instance based on provider with best available models."""

        if provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("Install langchain-openai: pip install langchain-openai")

            from langchain_openai import ChatOpenAI

            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found")

            return ChatOpenAI(
                model=model or "gpt-4o",
                api_key=key,
                temperature=temperature,
                max_tokens=4096
            )

        elif provider == LLMProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")

            from langchain_anthropic import ChatAnthropic

            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not found")

            return ChatAnthropic(
                model=model or "claude-sonnet-4-20250514",
                api_key=key,
                temperature=temperature,
                max_tokens=4096
            )

        elif provider == LLMProvider.GOOGLE:
            if not GOOGLE_AVAILABLE:
                raise ImportError("Install langchain-google-genai: pip install langchain-google-genai")

            from langchain_google_genai import ChatGoogleGenerativeAI

            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GOOGLE_API_KEY not found")

            return ChatGoogleGenerativeAI(
                model=model or "gemini-2.0-flash",
                google_api_key=key,
                temperature=temperature,
                max_output_tokens=4096,
                convert_system_message_to_human=True
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")
