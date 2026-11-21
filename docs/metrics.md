# The metrics

`uv run lfe evaluate runs/<run> [runs/<other run>]` compares what your extractor showed, and when, with the gold.
It writes `results/<name>/`: `summary.csv` (one row per run), `ranking.csv`, tables per field, per field type,
per conversation and per value, and `report.html`.

## What decides the ranking

**The score**: one number between 0 and 1, the harmonic mean of four terms.

| Term | What it asks | Read from |
|---|---|---|
| **Completeness** | Of what was said, how much did your form show? | every gold value |
| **Precision** | Of the time your form showed something, how much of it was right? | every display, by its seconds on screen |
| **Speed** | How soon after a value was said did your form show it? | every gold value |
| **Outcome** | Is the form right at the end? | the final form |

    score = 4 ÷ (1/completeness + 1/precision + 1/speed + 1/outcome)

The harmonic mean stays close to the weakest term: a form that is fast but wrong, or right but slow, scores
low. Each term is the mean over your forms; a submission with one run per dataset gets the average of its two
scores (weights in the config).

- **Official runs only.** The config's sample and stream settings, in real time. The organisers score every
  run with their own `config.yaml`; a run needs no new replay when the scoring changes.
- **Price is not in the score.** It is shown next to it, and the report plots one against the other: the upper
  left (a high score for a cheap form) is where to be.

Every other number is there to understand a result, not to rank it.

## Whose clock?

Your extractor's. The datasets time each word when it was spoken, but your extractor can only react once the
stream has delivered it. So a value's clock starts **when the words that say it reach your extractor**: the
arrival of the first partial or finished turn that holds them. The time the speech-to-text stream takes to
deliver a word is not counted against you. Below, "said" always means that moment.

## When is "the end"?

- **SpokenWOZ:** the form is read at `T_END` = the arrival of the last message, or 8 s (`score.speed_limit_s`)
  after the last value if that is later. Later than that, the last value could earn no speed anyway; what you
  publish after `T_END` is not counted.
- **AMI:** the form is read when your `run()` returns, at most 15 s (`final_grace_s`) after the last partial.

## What a field should show: the gold steps

The gold of a field is a list of values, each with the moment it was said. It follows the transcript word by
word: a dictated number grows digit by digit, a name word by word, a time reading by reading ("1:30" is 01:30
until "pm" makes it 13:30). A choice, a yes or no and a count are one piece of information each.

Your extractor never sees half a message. So the gold keeps **one step per stream message that changes the
field**: the value as it stands once that message has arrived.

    words          0 1 5   6 2 3                     spoken one after the other
    messages       (0 1)  (5 6 2)  (3)               as the stream delivered them
    gold steps     "01"   "01562"  "015623"

Between two steps there is nothing a form could have shown.

A note or an answer (AMI) is a set of **facts**. A fact counts from the message that completes it. When the
meeting settles a point and changes it later, the earlier version is in the gold until the message that changes
it. So the gold steps of a free-text field are the facts that hold after each message that starts or ends one.
[data.md](data.md) says how values are cut and how facts are made.

## The score, step by step

Everything below happens per field of one conversation, with two lists: the field's gold steps, and your
**displays**: every visible change of the field's value. Sending the value that is already there is not a
display; clearing the field is one.

### A value already on screen when it is said

If your form already shows a gold value at the moment it is said, it counts as displayed at that moment. This
rewards a good guess, or a name you completed from its first words: the seconds before it was said keep their
own grade (below), and from then on the value is there, with no delay.

### The walk: where is the form in the gold?

The displays are read once, in time order. `at` is the gold value the form has reached.

| A display… | is a | Grade |
|---|---|---|
| equals a gold value further than `at` that was already said | **landing**: `at` moves there (to the most recent one, if several fit) | 1 |
| equals the value at `at` again | **repair**: you came back to it after showing something else | 1 |
| anything else | – | its similarity (below) to the closest gold value from `at` on that was already said |

- **A value shown late is still right.** Lateness is charged once, in speed.
- **A value shown early is graded twice.** Until it is said, against what had been said so far; from then on
  as a landing.
- **Going back is wrong.** "Jacques" after "Jacques Ung" was shown gets the grade of a wrong "Jacques Ung".
- **A field nobody talked about has no gold value.** Whatever it shows has grade 0.

### The match: one display for every gold value

A gold value with a landing takes that display and scores 1. The others take their **closest display**:

- Gold values without a landing are taken run by run (a run = neighbours without a landing).
- Their displays are looked for from the landing before the run to the landing after it, both included (or
  from your first display, or to your last one).
- Only a display shown after the gold value was said can be taken.
- The run is read from its last gold value to its first, and a gold value never takes a display that comes
  after the one taken by the gold value that follows it. So the displays keep the order of the gold.
- A gold value takes the most similar display that is allowed (the earliest if two are equal), and scores that
  similarity. With nothing similar, it scores 0 and has no display.

### Similarity

How close a display is to a gold value, from 0 to 1:

- **Written values** (text, phone, email, time): the pairs of consecutive characters they share, the start
  marked. "0172" has the pairs ^0, 01, 17, 72 and "0171" has ^0, 01, 17, 71: 3 shared, so
  2 × 3 ÷ (4 + 4) = 0.75. "Jacques" against "Jacques Ung" is 0.78; "Gonzalo" against "Moritz" is 0.
- **Choices and amounts** (select, checkbox, integer, slider): the same value or not.
- **Multiselect:** the same formula on the ticked options.
- **Free text:** a judge reads your text against the facts of the gold value ("Free text (AMI)" below). The
  similarity is the harmonic mean of `correct` and `complete`, and your text equals the gold value when both
  are 1.
- An empty field against a value: 0.

### The four terms

    completeness of a field = mean score of its gold values
    speed of a field        = mean of (1 − delay ÷ limit) ^ 0.5 over its gold values
    precision of a field    = Σ grade × seconds on screen ÷ Σ seconds on screen, over its displays

- The **delay** of a gold value runs from the moment it was said to the display the match gave it. The limit is
  8 s on SpokenWOZ and 80 s on AMI; from the limit on, or without a display, the speed is 0. The curve stays
  near 1 for short delays and falls steeply towards the limit (limit 8 s: 0.5 s → 0.97, 2 s → 0.87,
  5 s → 0.61, 7 s → 0.35). These numbers are in the config and are placeholders until they are set on real runs.
- **Seconds on screen** run until your next display of that field, or until the end. So a wrong value that
  stays weighs more than a flash.
- Per form, completeness and speed are the mean over the fields that have gold values, and precision the mean
  over the fields that displayed something.
- **Outcome** is the final F1 of the form (next section): 2 × correct ÷ (2 × correct + wrong + missing +
  invented).

### Three examples

A phone number shown in two pieces (one digit per message; the call ends at 60 s):

    gold   0 at 1 s · 01 at 2 s · 015 at 3 s · 0156 at 4 s · 01562 at 5 s · 015623 at 6 s
    form   015 at 3.5 s · 015623 at 6.5 s

| Gold value | Display it takes | Score | Delay |
|---|---|---|---|
| `0` | `015` | 2×1 ÷ (3+1) = 0.50 | 2.5 s |
| `01` | `015` | 2×2 ÷ (3+2) = 0.80 | 1.5 s |
| `015` | `015`, a landing | 1 | 0.5 s |
| `0156` | `015623` | 2×4 ÷ (6+4) = 0.80 | 2.5 s |
| `01562` | `015623` | 2×5 ÷ (6+5) = 0.91 | 1.5 s |
| `015623` | `015623`, a landing | 1 | 0.5 s |

Completeness 0.83, speed 0.90, precision 1. Showing every digit half a second after it is said gives 1, 0.97
and 1.

A wrong value in between (the call ends at 60 s):

    gold   Moritz at 10 s
    form   Moritz at 11 s · Gonzalo at 20 s · Moritz at 50 s

Completeness 1 (a landing, 1 s late). Precision (9 s × 1 + 30 s × 0 + 10 s × 1) ÷ 49 s = 0.39: the second
"Moritz" is a repair.

A value shown before it is said:

    gold   Moritz at 10 s
    form   Moritz at 2 s

It counts from 10 s on, with no delay: completeness 1, speed 1. Precision 50 ÷ 58 = 0.86: for the 8 s before
it was said, nothing supported it.

### The algorithm

```python
# for one field of one conversation
# gold     = [(said, value), ...]    one value per stream message that changes the field
# displays = [(shown, value), ...]   every visible change of the field; a clear has an empty value
# same(a, g)        the field type's comparison (see "The final form")
# similarity(a, g)  0–1, as above

# 0. already on screen when it is said
for g in gold:
    d = the display on screen at g.said
    if d and same(d.value, g.value):
        insert a display (g.said, d.value)

# 1. the walk
at = None
for d in displays:                                   # in time order
    ahead = [g for g in gold after at if g.said <= d.shown and same(d.value, g.value)]
    if ahead:
        at = the most recent of ahead;  at.landing = d;  d.grade = 1
    elif at is not None and same(d.value, at.value):
        d.grade = 1
    else:
        owed = [g for g in gold from at on if g.said <= d.shown]
        d.grade = max(similarity(d.value, g.value) for g in owed) if owed else 0
    d.seconds = (shown time of the next display, or the end) - d.shown

# 2. the match
for g in gold with a landing:
    g.display, g.score = g.landing, 1
for run in runs of neighbouring gold values without a landing:
    window = displays from the landing before the run (else the first display)
             to the landing after it (else the last display), both included
    limit = None
    for g in run, from the last to the first:
        allowed   = [d for d in window if d.shown >= g.said and (limit is None or d.shown <= limit.shown)]
        g.display = the display of allowed most similar to g.value     # the earliest if equal; None if the best is 0
        g.score   = similarity(g.display.value, g.value) if g.display else 0
        if g.display: limit = g.display
for g in gold:
    g.delay = g.display.shown - g.said if g.display else None

# 3. the terms
completeness = mean(g.score for g in gold)
speed        = mean((1 - g.delay / LIMIT) ** POWER if g.delay is not None and g.delay < LIMIT else 0 for g in gold)
precision    = sum(d.grade * d.seconds for d in displays) / sum(d.seconds for d in displays)
```

Step 0 only puts displays in; steps 1 and 2 are the same with or without it. The code is
`live_form_eval/evaluation/timeline/walk.py` and one file per term in `evaluation/metrics/`.

## The final form

At the end, every unit of the form gets one outcome. A unit is a typed field's value, one option of a
multiselect field, or one fact of a free-text field.

| Outcome | Meaning |
|---|---|
| correct | The field shows the gold value. |
| wrong | It shows another value. |
| missing | It shows nothing; or the fact isn't in the text. |
| invented | It shows something where nothing belongs. |
| correctly empty | Nothing belongs there and nothing is shown. |

**What counts as the gold value** depends on the type: names match from 85% similarity ("acorn guesthouse" for
"acorn guest house"), identifiers and options exactly, phone numbers by their digits, times as HH:MM. Free text
is judged (below).

**Final F1**

    precision = correct ÷ (correct + wrong + invented)        everything you filled
    recall    = correct ÷ (correct + wrong + missing)         everything that should be filled

Correctly empty fields are left out on purpose: a SpokenWOZ form has 37 fields and about 28 stay empty, so
counting them would hide the errors. **Outcome** is this F1 computed per form and averaged over the forms;
`final_f1` in the tables adds the counts of all forms first.

**Fixes per form** = missing + wrong + invented, per conversation: what a user corrects by hand.

**Invented rate** = invented ÷ (invented + correctly empty): how often a field nobody talked about gets a value.

## Final Decision latency

For every unit that has a gold: from the moment its words reached your extractor until your form showed it
**for good** (it stays right until the end).

- A unit that isn't right at the end never settled: its latency is infinite.
- A unit shown before its words arrived has latency 0.
- Reported: the median, the 90th percentile, and the share that never settled. Infinite latencies take part in
  the percentile: if more than 10% of your values never settle, your p90 is infinite.

This is your extractor's own delay: waiting for its next model call, and the call. The time the stream took to
deliver the words is left out; `stream_delay_median_s` in `summary.csv` shows it for information. Latency is
reported, not ranked: speed is the term that counts.

## Joint goal accuracy and turn slot F1 (SpokenWOZ)

SpokenWOZ's own measures, for orientation only. After each caller turn, the dataset labels every slot. The form
is read 3 s after the turn's last words arrived; a turn counts for **JGA** when the whole form is right. **Turn
slot F1** counts the slots one by one.

Published trackers read finished turns without a time limit, and most are trained on SpokenWOZ: the paper's best
reached 25.65% JGA, ChatGPT 13.75%. Use these numbers as an orientation, not as a comparison.

## Free text (AMI)

Notes and answers are reworded freely, so they are **judged**: a natural-language-inference model decides whether
a text supports a statement.

### What the gold holds

- **Facts.** Each question's answer is a list of single statements of the meeting ("The Project Manager asked
  whether the remote controls should maybe be waterproof."), each with the moment the meeting has said all of
  it. These are the statements two independent readers both require in the answer
  ([data.md](data.md), "The answer key").
- **Accepted statements.** Statements an answer may also make without leaving the question (a reason, an
  objection, who agreed). They are never owed, and a line that says one is not invented.
- **Fields that must stay empty.** A question the meeting does not answer has no facts. Anything written there
  has grade 0 and is invented at the end; `unanswerable_accuracy` is the share left empty.

### How your text is compared with the facts

Your text is cut into units: the items of a list (one per line), or the sentences of a text.

| | Question | Result |
|---|---|---|
| complete | Is each fact supported by your whole text? | facts supported ÷ facts |
| correct | Is each of your units supported by the facts and the accepted statements together? | units supported ÷ units |

The similarity is the harmonic mean of the two, and your text equals a gold value when both are 1: it says
every fact, and nothing that the facts don't say. With that, a free-text field goes through the walk like any
other field.

    gold   "The remote is yellow." at 10 s · that and "It has a wheel." at 40 s
    form   both lines at 15 s, kept until the end at 200 s

Until 40 s the text says one thing too many: complete 1, correct 1 ÷ 2, similarity 0.67. From 40 s on it is the
gold value, with no delay. Completeness (0.67 + 1) ÷ 2 = 0.83, precision (0.67 × 25 s + 1 × 160 s) ÷ 185 s = 0.95.

### Which of your texts are judged

Every comparison costs judge calls, so a field may have one text judged per 30 s of the meeting
(`judge.interval_s`): 27 texts in a meeting of 13.5 minutes.

- A field that changes no more often than that has **every text judged**, each from the moment it was published.
- A field that is rewritten more often is only looked at every 30 s and at the end. A text that was replaced
  before the next look is not seen, and the text before it counts for those seconds.

So change a note when there is something new to say, not with every model call.

### At the end

The final text is compared with the facts that hold at the end. A fact your text supports is correct, the others
are missing, and each of your units that the facts don't support is invented.

    free-text F1 = harmonic mean of  correct (your units supported ÷ your units)
                                and  complete (final facts supported ÷ final facts)

A fact's latency runs from the moment it was said to the first judged text from which it stays supported until
the end.

### What the judge can miss

The judge reads a text against one statement at a time. It recognises the gold's own sentences 99.7% of the
time, and can miss a heavy paraphrase or a converted number ("1 g" for "1000 mg"). Where it misses, a right text
is a close display instead of a landing: its similarity is below 1, and nothing falls to 0.

**Long texts.** The model reads 512 tokens per question (about 2,400 characters of text and statement together)
and would drop the end of a longer text, so a fact stated near the end of a long note could not be found. A
text that is too long is therefore read in windows: as many of its sentences (or lines) as fit, each window
starting half a window after the one before (`judge.window_step`), and the text says a statement when one of
its windows does. Short texts are asked as one question, as before. Every window is one more chance of a wrong
"yes", so a text of 600 tokens (two windows) is judged a little more generously than one of 400.

**Topics** (a multiselect) need no judge: the field's gold steps are the topics named so far.

## Price and reliability

- **Price per form** and **per audio minute**: the euros of everything your extractor reported with
  `session.report_usage`, its model calls to fill the form and its other uses: for each use, the sum of amount ×
  price over its units ([interface.md](interface.md)).
- **Failure rate**: the share of conversations where `run()` raised an error or did not return in time.

## Model calls and value changes

Reported next to the score, to show where the price comes from and how the form behaved while people were
talking. None of it is ranked.

### Model calls, other uses and the price by unit

From the uses you reported with `session.report_usage` (`metrics/calls.py`). A use reported with
`extraction=True` is a **model call**: an attempt to fill the form. One reported with `extraction=False` is an
**other use** (a model that decides whether to extract now, …): it counts into the price, and in nothing else.

| Number | What it is |
|---|---|
| Calls per form, per audio minute | How often you call a model to fill the form. |
| Calls that changed a value | The share of model calls with a change of a field published nearest to them: every change goes to the model call reported closest to it in time, never to an other use. The other calls cost the same and changed nothing on screen. |
| A call takes | The median and the config's percentile (`latency.percentile`) of the `latency_s` of your model calls. |
| Tokens by kind | Read from cache (`cache_read_tokens`), fresh input (`input_tokens`), written to cache (`cache_write_tokens`), output (`output_tokens`), of your model calls. |
| Input read from cache | Tokens read from cache ÷ all input tokens. |
| Failed calls | Model calls reported with `ok=False` or with an `error`. |
| Other uses, their price | How many other uses a form had, and what they cost. |
| Price by unit | What each unit cost, over every use: the four kinds of tokens and any unit of your own (`audio_seconds`, …). |

A run made before units had prices reported each call's total in euros, its tokens with `input_tokens` counting
all input, and nothing else: its uses all count as model calls, its fresh input is `input_tokens` minus the cached
parts, and its price shows as one unit, "not split by unit".

### Value changes

Every visible change of a field is a **first fill** (the field was empty before), a **revision** (it held
another value) or a **clearing** (`metrics/value_changes.py`):

| Number | What it is |
|---|---|
| Value changes per form | All three kinds together. |
| Revisions per filled field | Revisions ÷ fields that were ever filled. 0 = every value was found once and kept. |
| Filled once, never changed | The share of the filled fields that changed a single time. |

A revision isn't an error by itself: a number dictated digit by digit and a note that grows are revised as they
should be. Precision says whether what stood on screen was right; these numbers say how restless the form was
to look at.

## Where to look

| File | What it holds |
|---|---|
| `summary.csv` | Every metric above, one row per run. |
| `ranking.csv` | One row per submission: its score over the datasets it ran, the four terms, the price per form, official or not. |
| `per_field_type.csv`, `per_field.csv` | Outcomes, final F1, completeness, precision, speed, latency and value changes by field type and by field. |
| `per_conversation.csv` | The same per conversation, with its score, its price, its model calls, its other uses and its tokens. |
| `values.csv` | Every unit of the final form: gold, shown, outcome, when it was spoken (`spoken_s`) and where that time comes from, when its words reached the extractor (`available_s`), and its latency. |
| `calls.csv` | Every use: when it was reported, whether it was a model call to fill the form (`extraction`), its price, how long it took, how many values changed with it, then one column per unit with its amount, and one per unit with its price (`eur_<unit>`). |
| `report.html` | The ranking and the score against the price, then one section per dataset with all its runs in the same tables and plots: scores, field types, model calls, tokens and the price by unit ("Where the money goes"), value changes. A run keeps one colour on every page. At the end, every form with each run's score on it. |
| `forms/<dataset>-<conversation>.html` | One page per conversation, chosen at the top of the report. The scores of each run on that form. A timeline: grey is what a field should show (one bar per gold step); green, amber and red are what a run showed, by grade (right, close, wrong), with a mark at every change of the field (● first fill, ▲ revision, ✕ cleared); rest the pointer on a bar or a mark to read its value in full, in a window that stays open while the pointer is in it and that scrolls. The model calls and the price on that form: a table of each run's calls and other uses and one of its price by unit; the price so far, one plot per run, with the whole price in the run's colour and each unit's price in grey, and a mark per use (● a model call that changed a value, ○ one that changed nothing, □ another use); the tokens of every model call by kind; model calls and value changes so far. The form at the end next to the gold (a note: "Extractor output 17 facts, gold has 16: 6 correct, 11 not, 10 missing"). **How each value was matched**: one row per gold value of each field with the display the scoring gave it for each run (shown, already on screen, the closest one, or nothing), its score and its speed: a field's completeness is the mean of those scores and its speed the mean of those speeds. Every field's gold steps and displays over time; a note's gold is listed fact by fact, with the moment each run's answer first holds each fact. The conversation as the extractor received it, as a table: what arrived, the gold those words complete, and what each run filled before the next line; one line per turn, or one per message (every partial and finished turn), which shows why a value that is dictated piece by piece has a gold step per message. The form itself: its rules and all its fields. |

Several runs named in one `lfe evaluate` (run folders, or the results folders of earlier evaluations) share one
report: one ranking, and every table, plot and form page shows them together, each run in its colour. It goes to
`results/comparison/` when the runs have several names (or where `--out` says).

The judge's verdicts on free text are kept in `results/.cache/`, one file per judge. A text pair is judged once:
evaluating a run again, or with other runs, asks the judge only about texts it has not seen. The folder can be
deleted at any time; the next evaluation then judges everything again and gets the same verdicts.

To watch one conversation instead of reading its page, `lfe ui` plays it: the form filling, and the score at
every moment ([ui.md](ui.md)). That score so far is the score of this page computed as if the conversation had
ended at that moment (only what was said and shown by then counts); at the end of the scoring it is the form's
score in `per_conversation.csv`.
