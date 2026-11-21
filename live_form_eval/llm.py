"""Text-in, text-out access to an LLM for the organisers' tools (the said-time matcher, SpokenWOZ's label check,
AMI's answer key, the optional LLM judge).

Three ways to run a model, behind one interface:

- ``MlxBackend``: an open model in this process, on Apple Silicon (the ``mlx`` extra).
- ``EndpointBackend``: any OpenAI-compatible server (vLLM, Ollama, llama.cpp, LM Studio), e.g. on a remote GPU.
- ``ClaudeBackend``: Claude through the Anthropic API (the ``claude`` extra).

Extractors don't use this module: a submission talks to its own models however it likes.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

Progress = Callable[[int, int], None]
_HTTP_TIMEOUT_S, _HTTP_ATTEMPTS = 180, 4


class Backend(Protocol):
    name: str  # recorded with what the model produced

    def complete(self, prompts: Sequence[str], max_tokens: int, answer_start: str = "",
                 schema: dict[str, Any] | None = None, progress: Progress | None = None,
                 shared: str = "") -> list[str]:
        """One reply per prompt (each prompt is a single user message).

        ``answer_start`` is how every reply should begin and ``schema`` the JSON schema it should follow. A
        backend uses the one it supports (MLX starts the reply for the model; Claude constrains the output to
        the schema) and ignores the other, so callers still parse replies tolerantly.

        ``shared`` is text that every prompt of the call starts with (the task, a long transcript). A backend
        that can stores it once and reads it from its cache for the other prompts (Claude); the others just
        put it in front of each prompt.
        """


def _whole(shared: str, prompt: str) -> str:
    return f"{shared}\n\n{prompt}" if shared else prompt


class MlxBackend:
    """An open model in this process, on Apple Silicon."""

    def __init__(self, model: str, batch: int) -> None:
        try:
            from mlx_lm import load
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("the MLX backend needs Apple Silicon and the mlx extra (uv sync --extra mlx); "
                               "on another machine, use --endpoint or --claude") from error
        self.name = f"mlx:{model}"
        self._batch = batch
        self._model, self._tokenizer = load(model)

    def complete(self, prompts: Sequence[str], max_tokens: int, answer_start: str = "",
                 schema: dict[str, Any] | None = None, progress: Progress | None = None,
                 shared: str = "") -> list[str]:
        from mlx_lm import batch_generate

        start = self._tokenizer.encode(answer_start, add_special_tokens=False) if answer_start else []
        tokens = [list(self._tokenizer.apply_chat_template([{"role": "user", "content": _whole(shared, prompt)}],
                                                           add_generation_prompt=True, tokenize=True,
                                                           enable_thinking=False)) + start for prompt in prompts]
        replies: list[str] = []
        for k in range(0, len(tokens), self._batch):
            texts = batch_generate(self._model, self._tokenizer, prompts=tokens[k: k + self._batch],
                                   max_tokens=max_tokens).texts
            replies += [answer_start + text for text in texts]
            if progress:
                progress(len(replies), len(tokens))
        return replies


class EndpointBackend:
    """An OpenAI-compatible chat endpoint: ``<url>/chat/completions`` (the url ends in /v1 on most servers)."""

    def __init__(self, url: str, model: str, api_key: str | None = None, parallel: int = 8,
                 extra_body: dict[str, Any] | None = None) -> None:
        self.name = f"endpoint:{model}"
        self._url = url.rstrip("/") + "/chat/completions"
        self._model = model
        self._headers = {"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
        self._parallel = parallel
        self._extra = extra_body or {}

    def _one(self, prompt: str, max_tokens: int) -> str:
        body = {"model": self._model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": max_tokens, **self._extra}
        request = urllib.request.Request(self._url, data=json.dumps(body).encode(), headers=self._headers)
        for attempt in range(_HTTP_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
                    return json.load(response)["choices"][0]["message"].get("content") or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                if attempt + 1 == _HTTP_ATTEMPTS:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def complete(self, prompts: Sequence[str], max_tokens: int, answer_start: str = "",
                 schema: dict[str, Any] | None = None, progress: Progress | None = None,
                 shared: str = "") -> list[str]:
        replies: list[str] = []
        with ThreadPoolExecutor(max_workers=self._parallel) as pool:
            for reply in pool.map(lambda prompt: self._one(_whole(shared, prompt), max_tokens), prompts):
                replies.append(reply)
                if progress:
                    progress(len(replies), len(prompts))
        return replies


class ClaudeBackend:
    """Claude through the Anthropic API, with the official SDK.

    Credentials are the SDK's own: ``ANTHROPIC_API_KEY`` or a profile from ``ant auth login``. The reply is
    constrained to the caller's JSON schema (these models take no pre-started reply). Thinking is always on in
    the current Opus and Fable models; ``effort`` sets how much, and ``max_tokens`` leaves room for it.

    A call's ``shared`` text is marked for the prompt cache: the first request stores it (billed a quarter
    above the input price), and requests sent within five minutes of it read it at a small fraction of that price.

    ``usage`` adds up what the requests consumed, so a run can print its real size. ``input_tokens`` are the
    tokens billed at the full input price; cached text is counted in ``cache_write_tokens`` and
    ``cache_read_tokens``.
    """

    # Models whose safety classifiers can decline a request: for these the API re-runs a declined request on
    # another model inside the same call, so one decline doesn't leave a hole in the labels.
    _WITH_FALLBACK = ("claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str, effort: str, max_tokens: int, parallel: int = 8, client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("the Claude backend needs the claude extra: uv sync --extra claude") from error
            client = anthropic.Anthropic(max_retries=5)  # a bulk job: ride out rate limits and overloads
        self.name = f"claude:{model}"
        self._client, self._model, self._effort = client, model, effort
        self._max_tokens, self._parallel = max_tokens, parallel
        self._lock = threading.Lock()
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0,
                      "cache_read_tokens": 0, "refusals": 0, "cut_off": 0, "failed": 0}

    def _one(self, prompt: str, max_tokens: int, schema: dict[str, Any] | None, shared: str) -> str:
        import anthropic

        output_config: dict[str, Any] = {"effort": self._effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        content: Any = prompt
        if shared:  # the cache holds everything up to the mark: the shared text, not the prompt after it
            content = [{"type": "text", "text": shared, "cache_control": {"type": "ephemeral"}},
                       {"type": "text", "text": prompt}]
        request: dict[str, Any] = {"model": self._model, "max_tokens": max(self._max_tokens, max_tokens),
                                   "messages": [{"role": "user", "content": content}], "output_config": output_config}
        try:
            if self._model in self._WITH_FALLBACK:
                response = self._client.beta.messages.create(**request, betas=[self._FALLBACK_BETA],
                                                             fallbacks="default")
            else:
                response = self._client.messages.create(**request)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                anthropic.BadRequestError):
            raise  # wrong key, model or request: every other request would fail the same way
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError):
            with self._lock:  # still failing after the SDK's retries: leave this one unanswered
                self.usage["requests"] += 1
                self.usage["failed"] += 1
            return ""
        text = next((block.text for block in response.content if block.type == "text"), "")
        with self._lock:
            self.usage["requests"] += 1
            self.usage["input_tokens"] += response.usage.input_tokens
            self.usage["output_tokens"] += response.usage.output_tokens
            self.usage["cache_write_tokens"] += getattr(response.usage, "cache_creation_input_tokens", None) or 0
            self.usage["cache_read_tokens"] += getattr(response.usage, "cache_read_input_tokens", None) or 0
            self.usage["refusals"] += response.stop_reason == "refusal"
            self.usage["cut_off"] += response.stop_reason == "max_tokens"
        return "" if response.stop_reason in ("refusal", "max_tokens") else text

    def complete(self, prompts: Sequence[str], max_tokens: int, answer_start: str = "",
                 schema: dict[str, Any] | None = None, progress: Progress | None = None,
                 shared: str = "") -> list[str]:
        replies: list[str] = []
        try:
            with ThreadPoolExecutor(max_workers=self._parallel) as pool:
                for reply in pool.map(lambda prompt: self._one(prompt, max_tokens, schema, shared), prompts):
                    replies.append(reply)
                    if progress:
                        progress(len(replies), len(prompts))
        except TypeError as error:  # how the SDK reports that it found no credentials
            if "authentication" not in str(error):
                raise
            raise RuntimeError("no Anthropic credentials found: set ANTHROPIC_API_KEY, or run `ant auth login`") from error
        return replies
