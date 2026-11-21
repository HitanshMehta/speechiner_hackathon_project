# The interface: what your extractor receives and what it must do

Your extractor fills a form while a conversation is going on. The harness replays the conversation as a live
transcript; you read it and publish values. **Nothing is returned**: you publish by calling the session, and the
harness records each call with the conversation's clock. Those records are what gets scored.

## The three methods

```python
from live_form_eval import Extractor, Session

class MyExtractor(Extractor):
    async def prepare(self, session: Session) -> None: ...   # optional
    async def run(self, session: Session) -> None: ...       # required
    async def close(self, session: Session) -> None: ...     # optional
```

| Method | Called | Timed? | What it does |
|---|---|---|---|
| `prepare(session)` | before the conversation starts | no | Optional. Build prompts from `session.form`, warm up a model. Publishing is refused here. |
| `run(session)` | when the conversation starts | **yes** | Reads `session.partials()`, publishes values, reports model calls. Returns after its last value. |
| `close(session)` | after `run` returned, failed, ran out of time or was stopped (the page's "Stop run", Ctrl+C on `lfe run`) | no | Optional. Frees what this conversation used: end here whatever `prepare` or `run` opened (a session, a connection), so that a stopped run leaves nothing open. Not called when `prepare` itself failed. |

One extractor object serves every conversation of a run, and several conversations run at the same time.
Keep a conversation's state **inside `run()`** (local variables), not on `self`. Shared things (an HTTP client,
a loaded model) belong on `self`, created in `__init__`.

`run` must return at most `final_grace_s` (15 s) after the last partial. Later, it is stopped and the
conversation counts as failed; what it published before still counts.

## The session

| Call | What it does |
|---|---|
| `session.form` | The form to fill (below). |
| `session.conversation_id` | The conversation's id. |
| `session.now()` | Seconds since the conversation started. |
| `async for p in session.partials()` | The live transcript: each `Partial` at the moment a speech-to-text service would send it. Ends after the last one. Iterate it once. |
| `session.update(field_id, value)` | Shows `value` in that field from now on. `None` clears the field. |
| `session.update_snapshot({field_id: value, …})` | Shows several values at the same moment. Fields that aren't in the dict keep their value. |
| `session.report_usage(model=…, extraction=…, usage={…}, prices={…}, latency_s=…)` | Reports one use of a model or a paid service: whether it was an attempt to fill the form, how much of each unit it used, and your vendor's price of one unit, in euros. Next section. |
| `await session.sleep(seconds)` | Waits in conversation time. Use it instead of `asyncio.sleep`, so a development run at `--speed 5` behaves like a real-time one. |

A value its field can't hold raises `live_form_eval.InvalidValue` and is not published (a whole snapshot is
refused if one of its values doesn't fit). The partials don't wait for you: if a model call takes 3 seconds, the
partials of those seconds are already there when you come back.

## Reporting what you use

Report every model call, and anything else you pay for, once it has answered. Your price is the sum of what you
report, and it stands next to your score.

```python
PRICES = {"input_tokens": 0.37e-6, "cache_read_tokens": 0.09e-6, "output_tokens": 1.47e-6}  # € per unit

session.report_usage(model="gpt-4.1-mini", extraction=True,          # a call that tries to fill the form
                     usage={"input_tokens": 344, "cache_read_tokens": 1200, "output_tokens": 34},
                     prices=PRICES, latency_s=0.81)
session.report_usage(model="gpt-4.1-nano", extraction=False,         # anything else: here, a model that only
                     usage={"input_tokens": 90, "output_tokens": 1},  # decides whether to extract now
                     prices=NANO_PRICES)
```

| Argument | Meaning |
|---|---|
| `model` | What you used: a model's name, a service. |
| `extraction` | `True` for an attempt to fill the form: a call whose answer may change its values. `False` for any other use: a model that decides whether it is worth extracting now, a summary you keep for later, a speech service. Only extraction attempts are your "model calls" in the report: calls per audio minute, tokens per call, calls that changed a value. Every use counts into the price. |
| `usage` | How much of each unit the use took, by the unit's name. The report knows four: `input_tokens` (input your vendor had not seen before), `cache_read_tokens` (input it read from its cache), `cache_write_tokens` (input it wrote to its cache) and `output_tokens`. Count each token once: OpenAI's `prompt_tokens` holds the cached part, so `input_tokens` is `prompt_tokens − cached_tokens`. Any other name is a unit too, made of letters, digits and `_`: `audio_seconds`, `requests`, `images`. |
| `prices` | Your vendor's list price of one unit, in euros, for every unit of `usage` (0 for a free one). A whole price list may be passed: units you didn't use are left out. The price of the use is the sum of amount × price. |
| `latency_s` | How long the use took. |
| `ok=False` or `error="…"` | A use that failed. It still costs what it used. |
| anything else | Kept with the use in the run folder (`trigger="partial"`). |

- **A vendor that bills a request as a whole** (a speech service, an API with a fixed fee): `usage={"requests": 1},
  prices={"requests": <its price in €>}`.
- **The report draws your price over time, unit by unit**, and splits it into what your model calls cost and what
  everything else cost ([metrics.md](metrics.md)).
- A unit without a price, a negative amount or `extraction` that is not `True` or `False` raises `ValueError` or
  `TypeError`: nothing of that use is kept.

## The live transcript

```python
Partial(t=7.99, kind="partial", turn_id="1", speaker="B",
        text="yes, here it is. how can i help.",
        words=(Word(text="yes,", start=5.21, end=5.47, speaker=None), …))
```

| Field | Meaning |
|---|---|
| `t` | When it arrives: seconds since the conversation started. |
| `kind` | `"partial"`: the turn so far. It **replaces** the previous partial of that turn; never append it. `"turn"`: the finished turn; its text no longer changes. |
| `turn_id` | Shared by a turn's partials and its finished turn. |
| `text` | The whole turn so far, or the finished turn. |
| `speaker` | The turn's main speaker so far: `"A"`, `"B"`, … in order of first speech. `"PENDING"` when there is under about a second of audio. |
| `words` | Each word with its audio `start` and `end`. In a finished turn each word has its own `speaker` (`"PENDING"` for a run of under a second, such as "yes."). In a partial the words have no speaker. |

A turn ends at a pause after a sentence, not when the speaker changes. So one turn can hold two speakers
(a question and its quick answer), and a long statement arrives as several turns. `p.labelled_text()` gives the
text with a speaker prefix at each change: `"B: yes, here it is. how can i help. A: yes, i'm looking for …"`;
`p.pieces()` gives the same as a list: `[("B", "yes, here it is. how can i help."), ("A", "yes, i'm looking for …")]`.

Words reach you about 1 s after they are spoken (0.3 s to 1.7 s), as from a real service. That delay is not
counted in your latency: your clock starts when the words arrive. [stream.md](stream.md) explains the timing.

## The form

```python
form.description      # the task rules: what to fill, and when. Read it; put it in your prompt.
form.fields           # the fields, in order
form.field("train_day")
```

Each field is a `FieldSpec`:

| Attribute | Meaning |
|---|---|
| `field_id` | The key you publish under. |
| `label` | A short name, e.g. "Train: day". |
| `prompt` | What the field means. |
| `type` | One of the 15 types below. |
| `options` | The allowed options of a `select` or `multiselect` field; `()` otherwise. |
| `range` | `(min, max)` of a `slider` field; `None` otherwise. |

## What each field type takes

| Type | Value to publish | Example |
|---|---|---|
| `text`, `textarea` | `str`. For a list field (its prompt says "one per line"), one item per line. | `"acorn guest house"` |
| `email`, `phone`, `url` | `str` | `"anna@example.com"`, `"01765550142"` |
| `integer` | `int` (not `"2"`, not `2.0`) | `2` |
| `float` | `int` or `float` | `89.5` |
| `slider` | a number inside the field's `range` | `4.0` |
| `checkbox` | `True` or `False`. Not filled is `None`: "no" and "not mentioned" are different. | `True` |
| `select` | exactly one of `options` (case matters) | `"north"` |
| `multiselect` | a `list` of `options` | `["train", "taxi"]` |
| `date` | `"YYYY-MM-DD"` | `"2027-06-05"` |
| `time` | `"HH:MM"`, 24-hour | `"15:30"` |
| `datetime` | `"YYYY-MM-DDTHH:MM"` | `"2027-06-05T15:30"` |
| `template` | a `dict` of sub-fields | `{"name": "Anna"}` |

`None` clears any field. The datasets use `text`, `textarea`, `email`, `phone`, `integer`, `slider`, `checkbox`,
`select`, `multiselect` and `time`; the other types are supported and covered by `lfe check`.

## What a run leaves behind

```text
runs/<run id>/
  run.json            the run: extractor, dataset, conversations and how each ended, speed, official?
  predictions.jsonl   {"conversation": "MUL2338", "t": 15.53, "field_id": "attraction_type", "value": "nightclub"}
  calls.jsonl         {"conversation": "MUL2338", "t": 15.40, "model": "…", "extraction": true,
                       "usage": {"input_tokens": 344, "cache_read_tokens": 1200, "output_tokens": 34},
                       "prices": {"input_tokens": 3.7e-07, …}, "eur": 0.000285, "latency_s": 0.81, "ok": true}
```

`lfe evaluate` reads only these files and the gold. That is also how Speechineer's own pipeline is scored: its
adapter implements the same three methods and writes the same folder.

## Before you run

`uv run lfe check my_extractor:MyExtractor` replays a short made-up call (two speakers, one field of every type)
twice at the same time and reports problems: an exception, a `run` that doesn't return, values a field can't
hold, no use reported (or none as a model call to fill the form), state shared between conversations.
