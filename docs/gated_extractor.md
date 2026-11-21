# Gated extractor

`my_extractor:MyExtractor` listens while one extraction request per conversation is in flight. Its local
gate prioritizes likely booking values and corrections. For meeting notes, it batches evidence over a longer
window. A deadline releases pending evidence even if the speaker pauses. Identical partials and finished turns
are coalesced after a successful update. If the unfinished turn produced no update, the finished turn can be
reviewed again. At stream end the extractor drains the pending evidence.

Requests contain the form schema, current populated fields, recent context and unreviewed transcript. OpenRouter
receives a structured JSON schema for the delta. Each change includes a short transcript quote. The extractor
checks the quote, field type and value before updating the form. SpokenWOZ's explicit service requests can fill
the service field locally. A station spelling variant is accepted only when it maps to one listed option.
AMI note changes add or remove individual fact lines and preserve the other lines.

## Offline check

From the repository root:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv run --locked --no-sync python -B -m scripts.check_gated_extractor
```

The smoke check injects an in-memory backend and feeds it the bundled transcript replays. It also checks
corrections, partial timing, final buffer handling, and note merges. Request counts measure scheduling.
Use the live pilot to measure accuracy and time to correct values.

## OpenRouter

Store `OPENROUTER_API_KEY` as an environment secret and allow outbound HTTPS to `openrouter.ai`.
OpenRouter's `GET /api/v1/key` endpoint can check access without making a completion.
The default model ID is `google/gemini-2.5-flash-lite`. It uses the OpenRouter key.

Before approving a live run, set the following non-secret variables. Confirm the route's current rates with
OpenRouter for the selected model.

| Variable | Units or behavior |
|---|---|
| `LFE_OPENROUTER_INPUT_USD_PER_M` | USD per million uncached input tokens |
| `LFE_OPENROUTER_OUTPUT_USD_PER_M` | USD per million output tokens |
| `LFE_OPENROUTER_EUR_PER_USD` | EUR per USD for benchmark cost reports |
| `LFE_OPENROUTER_CACHE_READ_USD_PER_M` | Optional fallback rate if a response includes cache reads without a billed total |
| `LFE_OPENROUTER_CACHE_WRITE_USD_PER_M` | Optional fallback rate if a response includes cache writes without a billed total |
| `LFE_OPENROUTER_RUN_BUDGET_USD` | Local spending limit shared across conversations in one process. Default `2`. |
| `LFE_OPENROUTER_ENABLE_PAID` | Set to `1` only for an authorized live run |

The extractor logs completed attempts and uses OpenRouter's billed total for cost when supplied. The configured
rates are used for reservations and as a fallback. A missing charge stops further requests when cost cannot
be computed. An interrupted request can still bill at OpenRouter. The local cap resets per process, so set
a provider-side key limit at the team's $20 ceiling for cumulative enforcement.

After live testing is authorized, start with `SNG0877`, the shortest bundled conversation at about 60 seconds.
The rates below were verified against OpenRouter's standard Gemini 2.5 Flash-Lite route. Set
`LFE_OPENROUTER_EUR_PER_USD` to the exchange rate used for the comparison before running the command.

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
export LFE_OPENROUTER_INPUT_USD_PER_M=0.10
export LFE_OPENROUTER_OUTPUT_USD_PER_M=0.40
export LFE_OPENROUTER_CACHE_READ_USD_PER_M=0.01
export LFE_OPENROUTER_CACHE_WRITE_USD_PER_M=0.0833333333333333
LFE_OPENROUTER_ENABLE_PAID=1 LFE_OPENROUTER_RUN_BUDGET_USD=0.05 \
  uv run --locked lfe run my_extractor:MyExtractor --dataset spokenwoz --ids SNG0877 --name gated-pilot
```

Set `PILOT_RUN` to the run folder printed by the CLI, then score the saved artifacts.

```bash
uv run --locked lfe evaluate speechineer/*-spokenwoz "$PILOT_RUN" --out results/gated-pilot-comparison
```

Compare the two `SNG0877` rows in `per_conversation.csv` for cost, call count, speed, final F1, and
final decision latency. Check `calls.csv` for failed calls and unknown charges. The report's summary rows
cover the whole saved baseline and the single pilot, so use the per-conversation comparison for the pilot.
The `lfe check` CLI runs the extractor and can make paid calls when the backend is enabled.
