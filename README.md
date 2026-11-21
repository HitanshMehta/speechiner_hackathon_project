# cf-hackathon-bench

An environment to build and score **live form extractors**: programs that fill a form while a conversation is
still going on, from the live transcript a speech-to-text service sends.

You write an extractor. The environment replays real conversations to it as a live stream, records what your
extractor shows in the form and when, and scores it: is the form right at the end, was it right while people
were talking, how long did each value take, and what did it cost.

Two datasets:

| Dataset | Conversations | Form |
|---|---|---|
| **SpokenWOZ** | phone calls to a booking service (41–452 s each): 6 calls | 37 typed fields (select, checkbox, integer, time, …), the same for every call |
| **AMI** | a design-team meeting of 14 minutes (1 meeting) | 6–14 questions about the meeting, each answered with a list of facts (who said what), and the topics covered; a form per meeting |

The organisers keep further conversations of both datasets that are not in this repository: the final
comparison of all submissions is run on them.

## Install, and get the data

```bash
git clone <this repo's URL> cf-hackathon-bench && cd cf-hackathon-bench
uv sync                # Python 3.12 and every dependency, into .venv
```

The data is in the repository, in `data/processed/` (where `config.yaml` looks for it): the conversations, their
forms and their gold. Using it means accepting the datasets' licences (`DATA_LICENSES.md`). Check the install with
`uv run lfe run examples.oracle_extractor:Oracle --dataset spokenwoz --n 2 --speed 30`: it must end
with a run folder.

## Write an extractor

An extractor is a class with one required method. The harness calls it once per conversation; you read the live
transcript and publish values by calling the session. Nothing is returned: the harness records every call with
the conversation's clock.

```python
from live_form_eval import Extractor, Session

class MyExtractor(Extractor):
    async def run(self, session: Session) -> None:
        turns, partial = [], ""
        async for p in session.partials():          # arrives in real time, as from a speech-to-text service
            if p.kind == "turn":                     # a finished turn
                turns.append(p.labelled_text())
                partial = ""
            else:                                    # the turn so far: replaces the previous partial
                partial = p.labelled_text()
            values, tokens = await my_model(session.form, turns, partial)
            session.update_snapshot(values)          # shown in the form from this moment
            session.report_usage(model="my-model", extraction=True,   # a call that tries to fill the form
                                 usage=tokens,       # e.g. {"input_tokens": 1544, "output_tokens": 30}
                                 prices=PRICES)      # your vendor's € per unit: {"input_tokens": 0.37e-6, …}
```

`examples/llm_extractor.py` is a complete, working extractor to start from; `my_extractor.py` is an empty one
with a place for your gating rule. For the LLM example, copy `.env.example` to `.env` (git ignores it), put your
key in, `uv sync --extra openai`, and start the commands with `uv run --env-file .env` instead of `uv run`.
`docs/interface.md` describes every object (`Session`, `Partial`, `Form`, `FieldSpec`) and the value each field
type takes.

## From your extractor to your results

```bash
uv run lfe check my_extractor:MyExtractor                                          # 1. contract check on a made-up call (seconds)
uv run lfe run my_extractor:MyExtractor --dataset spokenwoz --n 2 --speed 5           # 2. a quick development run
uv run lfe evaluate runs/<run id>                                                  # 3. the report: scores, plots, and a page for every form
# repeat 2–3 while you develop, then:
uv run lfe run my_extractor:MyExtractor --dataset spokenwoz                        # 4. official run: the config's sample, real time
uv run lfe run my_extractor:MyExtractor --dataset ami
uv run lfe evaluate runs/<spokenwoz run> runs/<ami run>                            # 5. your official results
```

A run writes `runs/<run id>/` (`run.json`, `predictions.jsonl`, `calls.jsonl`). An evaluation writes
`results/<name>/`, where `<name>` is your extractor's class name (or `--name`):

| Open | To see |
|---|---|
| `report.html` | The ranking and the score against the price, then one section per dataset: your score and its four terms, each field type (what the final form got right; completeness, precision and speed while people were talking), your model calls (how many, how many of them changed a value, their tokens by kind) and your other uses, your price unit by unit, and how often values changed. At the end, every form with its score. |
| A form's page: choose the form at the top of the report (the files are in `forms/`) | One conversation in full. Your scores on it. A timeline of what each field should have shown and what yours showed, coloured right, close or wrong, with a mark at every change; rest the pointer on a bar to read its value in full, in a window you can move into and scroll. Your model calls and other uses on it: the price so far, in all and unit by unit, the tokens of every call, calls and value changes over time. The form at the end next to the gold. How each gold value was matched: the display the scoring gave it, with the score and the speed that follow, so you can see where completeness and speed come from. Every field's values over time (a note fact by fact), the conversation as your extractor received it (turn by turn, or message by message) with the gold said and the values you filled at each line, and the form itself (its rules and all its fields). The fastest way to see what to fix. |
| `per_conversation.csv`, `per_field.csv`, `values.csv`, `calls.csv` | Which conversations and fields cost you most, every value of every final form next to its gold, and every use you reported with its units and its price. |

To compare runs (two versions of your extractor, or several models), name them together: run folders, or the
results folders of earlier evaluations. They are scored into one report: one ranking, and per dataset all runs
in the same tables and plots, each in its own colour; on every form's page, what each of them showed.

```bash
uv run lfe evaluate results/MyExtractor results/MyOtherExtractor     # → results/comparison/ (or where --out says)
```

## Compare with Speechineer

`speechineer/` holds Speechineer's official runs on the same conversations (the 6 SpokenWOZ test calls and the AMI
meeting): what its form showed and when, and what each of its model calls cost. They were made with this repository's harness and are scored by the same code as yours.
Name them next to your runs:

```bash
uv run lfe evaluate speechineer/*-spokenwoz speechineer/*-ami runs/<your spokenwoz run> runs/<your ami run>
```

The report (`results/comparison/`) ranks both, shows both in every table and plot, and on every form's page
what each of you showed and when.

## Watch one conversation

```bash
uv run lfe ui     # a page in your browser; start it in the folder that holds your extractor
```

The page plays one conversation and shows its form filling: the transcript as your extractor receives it, each
field with what it shows and what it should show, the score at every moment, the price, and the recording if
you turn the sound on. `docs/ui.md` describes it.

- **Live run:** choose a form and your extractor, press "Start run". Your extractor runs on that conversation
  now, at the speed you choose, so whatever it calls is really called. Your file is read again at every start.
- **Review a run:** choose a run you made before. It is replayed from its run folder, and nothing is called.

- **No GPU is needed.** Evaluating AMI is faster with one, because a language model reads the notes and
  answers: `lfe evaluate` uses a GPU when there is one. `docs/kaggle.md` shows how to do it on a free Kaggle
  notebook. What that model decided is kept in `results/.cache/`, so evaluating the same run again, or
  comparing it with another, takes seconds.
- **A run folder is all an evaluation needs.** `lfe evaluate` can score an old run again at any time, for
  example after the metrics have changed, without running the extractor.
- **The report is a folder.** `report.html` links to the pages in `forms/`: to pass a report on, send the
  whole `results/<name>/` folder. Its plots load their drawing library from the internet.
- **Runs are told apart by their name.** Give two versions of one extractor different names
  (`lfe run … --name v2`); two runs that share a name show their start time next to it.

## The rules

- **Official run:** no sampling option and no `--speed`. It uses the sample in `config.yaml` and replays in
  real time, so latency is real. Anything else is marked unofficial in the results. A run stays official when
  the way runs are scored changes (it is scored again); it stops being official when the official sample or
  the stream settings change.
- **`config.yaml` is the organisers' file:** it defines how every submission is evaluated. Don't change it.
  Everything you tune (prompts, models, how often you call them) lives in your own code.
- **Develop on few conversations, fast.** `--n 2 --speed 5` (or `--ids <id>`) keeps a development run short and
  cheap. The 6 calls and the AMI meeting are all you have: a strategy fitted to their details will not carry over
  to the conversations the final comparison uses.
- **Report what you spend:** every model call, and anything else you pay for, goes through
  `session.report_usage(…)` with your vendor's price of each unit. Say whether it was an attempt to fill the form
  (`extraction=True`) or anything else (`extraction=False`: a model that only decides whether to extract now, …).
  Price is shown next to your score, unit by unit.
- **The ranking:** one score, the harmonic mean of four terms. Completeness: was each value shown? Precision:
  was what stood on screen right? Speed: how soon after it was said? Outcome: is the form right at the end?
  `docs/metrics.md` explains every number.
- **What your form shows must come from your extraction.** Only a change of a field's value counts as a
  display; sending the same value again changes nothing.

## Where things are

```text
config.yaml            how submissions are evaluated (documented in docs/config.md)
docs/                  interface, stream, data, metrics, config, ui (the page), kaggle (where it runs: CPU, GPU,
                       Kaggle)
examples/              a complete LLM extractor and an oracle
notebooks/             02 the stream and a mock extractor · 03 evaluating a run
speechineer/           Speechineer's official runs, to compare with
live_form_eval/
  interface/           what you code against
  harness/             the simulated live stream, the clock, the runner
  data/                the data model, the store of the processed data, the recordings
  evaluation/          matching, timelines, one file per metric, the report
  ui/                  the page of `lfe ui`: its server and the page itself
```

## Data licences

SpokenWOZ is CC BY-NC 4.0 (non-commercial). AMI is CC BY 4.0. See `DATA_LICENSES.md` for all four sources, and
for the recordings that `lfe ui` fetches when you turn the sound on.
