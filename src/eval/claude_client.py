"""Shared Claude client helpers for the eval harness (generation + judging)."""

from __future__ import annotations

import os
import re

from anthropic import Anthropic, AsyncAnthropic
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("NUTRIGUARD_MODEL", "claude-sonnet-4-5-20250929")
# A cheaper/faster model is fine for judging; default to the same model
# family for consistency unless overridden.
DEFAULT_JUDGE_MODEL = os.environ.get("NUTRIGUARD_JUDGE_MODEL", "claude-haiku-4-5-20251001")


def require_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Create a .env file in the project root "
            "with ANTHROPIC_API_KEY=sk-ant-... or export it in your shell."
        )


def simple_generate(prompt: str, model: str = DEFAULT_JUDGE_MODEL, max_tokens: int = 1024) -> str:
    """One-off generation helper (used for synthetic query / golden question generation)."""
    require_api_key()
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _extract_json(text: str) -> str:
    """Strip markdown code fences if the model wrapped its JSON output in them."""
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


class ClaudeDeepEvalModel(DeepEvalBaseLLM):
    """Wraps Claude as a DeepEval judge model, so faithfulness/relevancy scoring
    uses the same LLM provider as the rest of this project (no implicit
    dependency on an OpenAI key, which DeepEval assumes by default).
    """

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL):
        self.model = model
        self.sync_client = Anthropic()
        self.async_client = AsyncAnthropic()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        require_api_key()
        response = self.sync_client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if schema is not None:
            return schema.model_validate_json(_extract_json(text))
        return text

    async def a_generate(self, prompt: str, schema=None) -> str:
        require_api_key()
        response = await self.async_client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if schema is not None:
            return schema.model_validate_json(_extract_json(text))
        return text

    def get_model_name(self) -> str:
        return f"Claude ({self.model}) via Anthropic API"
