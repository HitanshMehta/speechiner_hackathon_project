# The page: watch a form fill (`lfe ui`)

`lfe ui` opens a page that plays one conversation and shows its form filling: the transcript as an extractor
receives it, every field with what it shows and what it should show, the score at every moment, and the price.
It can play the recording along. Use it to see what your extractor does on one conversation before you run it on
many.

```bash
uv run lfe ui                      # opens http://127.0.0.1:8765/ in your browser
uv run --env-file .env lfe ui      # the same, when your extractor reads a key from .env
uv run lfe ui --port 8800 --no-open
```

Start it in the folder that holds your extractor. Ctrl+C stops it, and with it a live run that is still going.

## What you choose

| Control | What it does |
|---|---|
| **Live run** / **Review a run** | The two ways to play a conversation (next section). |
| **Form** | One conversation of the config's sample (`datasets` in `config.yaml`), with its length and the number of its fields that have something to show. When you review, the conversations of earlier runs are listed too. |
| **Extractor** (live) | Every extractor class found in the folder you started the page in, and in its subfolders: a class that subclasses `Extractor` and has a `run` method. The list is refreshed every two seconds, so a new file or class shows up by itself. A file with a syntax error is listed with its error. |
| **Run** (review) | The run folders in `runs/` that cover the chosen form, the newest first. "No run" plays the conversation alone. |
| **Speed** | 1× is real time. |
| **Sound** | Plays the conversation's recording along. |

## Two ways to play a conversation

### A live run

Your extractor is started on the chosen conversation, in a process of its own, and receives the live stream at
the chosen speed. **Whatever it calls is really called, and billed if it bills.**

- **Your file is read again at every start.** Edit it, press "Start run" again.
- **"Stop run" stops it.** So does closing the page (after five seconds) and stopping `lfe ui`: an extractor
  never keeps calling a model with nobody watching.
- **"Extractor output"** under the page holds what your extractor prints, and the error if it fails.
- **Speed:** a faster replay makes the conversation's clock run faster, not your model. At 5×, a call that takes
  one second answers five seconds of conversation later, so delays look five times as long. Use other speeds to
  see *what* gets filled, and 1× to see *how fast*.
- **It leaves a run folder** in `runs/`, like `lfe run … --ids <conversation>`: you can review it afterwards and
  score it with `lfe evaluate runs/<run id>`. A run of one conversation is never an official run.
- One live run at a time. Reloading the page stops the run.

### A review

A run folder is loaded and replayed: what the form showed, the model calls, the score. **Nothing is called**, so
it costs nothing. Any run folder works: a live run from the page, an `lfe run`, an official run.

- Play and pause, at any speed.
- Jump: drag the position, click a time in the conversation, or click one of the small graphs. The keyboard
  works too: space plays and pauses, the left and right arrows move five seconds.
- What lies after the current position is drawn faint, so you can see where something happens before you get
  there.

## What the page shows

### The conversation

On top, the message that arrived last, like a subtitle: the turn so far or the finished turn, with its new words
standing out. A message appears when a speech-to-text service would have sent it, about a second after the words
are spoken ([stream.md](stream.md)). That second is not a lag of the page: it is when your extractor gets the
words, and your clock starts there.

Under it, the conversation as a timeline with three columns:

| Column | Holds |
|---|---|
| What arrived | The words, with the speaker's letter. |
| Should be filled | The gold values those words complete: what a field should show from this line on. |
| Was filled | What the form filled while this line was the latest thing it had received, each with its mark. |

Two ways to cut it:

- **Turns:** one line per turn, growing while it is spoken.
- **Every message:** one line per partial and finished turn, exactly as the extractor got them. A partial carries
  its whole turn so far; the line shows the turn's last words in grey and the new ones in black.

"Every message" is the view that explains the gold of a value that is spoken piece by piece. A phone number
arrives in several messages, and each of them changes what the field should show: "01", then "01562", then the
whole number. That is why such a field has several gold values, and why a form that waits for the end of the
turn is late on the first ones.

"Just said" lists the values that were said in the last seconds: for as long as showing them still earns some
speed (`score.speed_limit_s`).

### The form

On top, the form's rules: the text an extractor gets for the whole form (`session.form.description`). Under it,
one card per field. "Fields in play" lists the fields something is said about and the fields the form filled;
"All fields" lists the whole form.

| On a card | Meaning |
|---|---|
| The line under its name | The field's prompt: what it asks for, as the extractor gets it (`spec.prompt`). With the rules, it says whether a value belongs in the field. |
| The value | What the field shows now. It lights up when it changes. |
| The mark and its line | How that compares with what was said (below). |
| The grey line | What the field should show, over the conversation's time: one bar per gold value. |
| The coloured line | What it showed: green = right, amber = close, red = wrong. |

| Mark | Meaning |
|---|---|
| ✓ Right | It shows the value that was said, with the seconds it took. |
| ≈ Close | It shows something close to it (a name with a missing word, a number with a missing digit). |
| ✕ Wrong | It shows another value. |
| … Waiting | A value was said and is not on the form yet, with the seconds since. After the speed limit, showing it earns no speed any more. |
| + Invented | It shows a value where the field should be empty: nothing was said for it yet, or what was said was withdrawn. The value's window says which, next to the field's prompt and the form's rules. |

- **A note or an answer** (AMI) shows its text sentence by sentence next to the facts it should state by now.
  A ✓ on a sentence: the facts said so far support it. A + on a sentence: they don't, so it counts as invented,
  even when it is true to the meeting (the judge reads the reference, not the meeting). A ✓ on a fact: the text
  says it. Facts that are said later are listed in grey with their time.
- **A value shown before its words arrived** ("2.1 s before it was said") is graded against what had been said
  until then, and counts as right, with no delay, from the moment it is said.
- **Rest the pointer on a value or on a bar** to open a window with all of it: the whole text, from when to
  when, the words that say it, the grade. The window stays open while the pointer is on the value or inside the
  window, so you can move into it and scroll.
- **Click a field's name** for every value it should have shown and every value it showed.

### The score so far

Six small graphs over the conversation's time: the score, its four terms, and the price.

**The score so far is the official score of this form, computed as if the conversation ended at that moment:**
only what was said and shown by then counts, and the form is read then ([metrics.md](metrics.md) explains the
terms). Two things follow:

- A value that was just said counts as missing until the form shows it. So the lines dip at every new value, for
  as long as your extractor takes, and deeper early in a conversation, when few values have been said.
- From the end of the scoring on, it is the official score of the form: the number `lfe evaluate` gives for it.

The price is what your extractor reported with `session.report_usage` so far: its model calls to fill the form and
its other uses, counted apart under the price ("57 model calls so far · 3 other uses"); the small marks under its
line are the uses. Rest the pointer on a graph to read all six at that moment, with the price unit by unit;
"Show as a table" lists them.

Free text needs the judge, a language model that reads the notes ([kaggle.md](kaggle.md)). Its verdicts are kept
in `results/.cache/`, so a run that was evaluated before is replayed from them. The first time a text pair comes
up that nobody judged yet, the model is loaded: up to a minute and about 2 GB of memory, once.

### The sound

The data holds no audio. When you turn the sound on, the recording of that one conversation is fetched and kept in
`paths.audio`: about 3 MB for a call, 27 MB for a meeting.

- **SpokenWOZ:** only the test split can be fetched call by call. The dev recordings come as one archive of
  10 GB; to hear a dev call, save its recording as `<paths.audio>/spokenwoz/<conversation id>.wav`.
- **Sync:** in a review, the recording leads the clock, so what you see and what you hear can't drift apart. In a
  live run, your extractor's clock leads and the recording is kept on it.
- **Speed:** the recording is played up to 4× (`ui.audio_max_speed`).
- **Licences:** SpokenWOZ's recordings are CC BY-NC 4.0, AMI's are CC BY 4.0 (`DATA_LICENSES.md`).

## Settings

The `ui` section of `config.yaml` ([config.md](config.md)): the address and port, where extractors are looked
for, the speeds on offer, how often the score so far is computed. None of them changes a score.

The page is served to this machine only (`127.0.0.1`). It answers no request from another site, and only the
page itself can start or stop a run.
