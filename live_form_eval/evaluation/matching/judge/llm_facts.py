"""The alternative judge for free text: an LLM checks one statement against a text.

Slower than the NLI judge and not the default. It is useful as a second opinion: an LLM reads paraphrases and
numbers ("1 g" = "1000 mg") that an NLI model sometimes misses. Set ``judge.kind: llm_facts`` and
``judge.llm_model`` in the config. The model runs with MLX on Apple Silicon, or behind an OpenAI-compatible
server when ``LFE_JUDGE_ENDPOINT`` is set (``LFE_JUDGE_API_KEY`` optional).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence

from live_form_eval.config import JudgeConfig
from live_form_eval.llm import Backend, EndpointBackend, MlxBackend

PROMPT = """Text:
{text}

Statement:
{statement}

Does the text say what the statement says? Answer "same" if it does (other words are fine), "wrong" if the text says something that contradicts it, "missing" if the text doesn't mention it.
Reply with JSON only: {{"verdict": "same" | "wrong" | "missing"}}"""
ANSWER_START = '{"verdict":'
ANSWER_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["same", "wrong", "missing"]}},
                 "required": ["verdict"], "additionalProperties": False}
_MAX_TOKENS = 16  # the answer is one word in a small JSON object


class LlmFactsJudge:
    def __init__(self, config: JudgeConfig, backend: Backend | None = None) -> None:
        if backend is None:
            if not config.llm_model:
                raise ValueError("judge.kind is llm_facts but judge.llm_model is not set")
            endpoint = os.environ.get("LFE_JUDGE_ENDPOINT")
            backend = (EndpointBackend(endpoint, config.llm_model, api_key=os.environ.get("LFE_JUDGE_API_KEY"))
                       if endpoint else MlxBackend(config.llm_model, config.batch))
        self._backend = backend
        self.name = f"llm_facts:{backend.name}"
        self._cache: dict[tuple[str, str], bool] = {}

    def supports(self, pairs: Sequence[tuple[str, str]]) -> list[bool]:
        new = sorted({pair for pair in pairs if pair not in self._cache and pair[0].strip() and pair[1].strip()})
        if new:
            prompts = [PROMPT.format(text=text, statement=statement) for text, statement in new]
            for pair, reply in zip(new, self._backend.complete(prompts, _MAX_TOKENS, ANSWER_START, ANSWER_SCHEMA)):
                self._cache[pair] = _verdict(reply) == "same"
        return [self._cache.get(pair, False) for pair in pairs]


def _verdict(reply: str) -> str | None:
    for candidate in re.findall(r"\{.*?\}", reply, re.DOTALL):
        try:
            return str(json.loads(candidate)["verdict"]).lower()
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return None
