# config.yaml, key by key

`config.yaml` defines **how submissions are evaluated**. The organisers set it, and score every run with it.
Nothing a submission tunes is in this file: prompts, models and call timing live in your own code.

A run stores the settings it was made with. Two kinds matter differently afterwards:

- **What a run cannot change later:** the sample of its dataset (`datasets`) and the `stream`. They decide which
  conversations the extractor got, and how. If they change, earlier runs are marked unofficial.
- **What is applied when a run is evaluated:** everything else (`score`, `matching`, `judge`, `latency`, and how
  the gold is built). If they change, `lfe evaluate` scores earlier runs again with the new settings, and they
  stay official. The report says so in a note: it compares the run's stored settings with today's, section by
  section, on the sections that decide a score.

The code holds no default for any of these keys: a missing key is an error.

## paths

| Key | Meaning |
|---|---|
| `raw` | The downloaded datasets the data is built from (about 490 MB). Outside the repo; only building the data uses it. |
| `processed` | The conversations, forms and gold: where they are built, and where every command reads them. |
| `audio` | Where `lfe ui` keeps the recordings it fetches, one per conversation that somebody played with the sound on (about 3 MB a call, 27 MB a meeting). Outside the repo. A recording saved there by hand as `<dataset>/<conversation id>.wav` is used as it is. |
| `runs` | One folder per run (`lfe run`, and a live run from the page of `lfe ui`). Relative paths are inside the repo. |
| `results` | What `lfe evaluate` writes: one folder per evaluation (tables, `report.html`, a page per form in `forms/`), and `.cache/`, where the judge of free text keeps its verdicts so that a text pair is only judged once (`docs/metrics.md`, "Where to look"). |

Paths are not part of the fingerprint.

## datasets: the official sample

| Key | Meaning | Effect |
|---|---|---|
| `spokenwoz.split` | `test` (this repository: 6 calls) | Official runs use `test`. The dev split is not shipped. |
| `spokenwoz.n` | Calls drawn from the split; `null` = all | 100 calls ≈ 4.4 h of audio; at 4 calls at a time, about 1.1 h per run. |
| `spokenwoz.seed` | Seed of the draw | The same `n` and `seed` always give the same calls. |
| `ami.max_minutes` | Longest meeting kept | 15 keeps 9 of the 142 summarised meetings (1.8 h). 20 would keep 24, 30 would keep 54. |
| `ami.n`, `ami.seed` | As above, among the meetings kept | `null` = all 9. |
| `ami.layers` | Which label layers become form fields: `summary`, `topics`, `qmsum`, `mised` | Removing a layer removes its fields from every meeting's form, when the data is built. |

## stream: the simulated speech-to-text stream

Measured on four real AssemblyAI (u3-rt-pro) sessions recorded through Speechineer's engine; see
[stream.md](stream.md) for the measurements behind each value.

| Key | Value | Meaning |
|---|---|---|
| `partial_every_s` | 1.3 | While a turn has new words, a partial arrives about this often. Measured: median 1.21–1.40 s. |
| `partial_jitter_s` | 0.2 | Each interval varies by ± this much. Measured: p10 ≈ 1.1 s, p90 ≈ 1.6 s. |
| `word_delay_s` | 0.4 | A partial shows the words that ended at least this long ago. With the interval above, a word appears 0.9–1.0 s after it was spoken on median (measured: 0.75–1.06 s). |
| `turn_pause_s` | 0.2 | After a word that ends a sentence, a silence this long ends the turn. Fitted: the rule predicts 102 turn ends where the vendor made 107. |
| `turn_max_silence_s` | 1.5 | A silence this long ends the turn anywhere. The vendor's own mechanism; no recording had such a pause mid-sentence. |
| `turn_close_s` | 0.3 | The finished turn arrives this long after its last word. Measured: median 0.27 s. Must be ≥ `turn_pause_s`. |
| `max_turn_s` | 60 | Continuous speech is cut into turns of at most this length. Measured on monologues: a turn every 60 s. |
| `speaker_labels` | true | Speakers "A", "B", … on turns and on the words of finished turns. `false` gives the stream of an adapter without speaker labels. |
| `pending_under_s` | 1.0 | Under this much audio, a speaker is "PENDING". From AssemblyAI's documentation ("less than approximately 1 second"). |
| `final_grace_s` | 15 | After the last partial, an extractor has this long to return from `run()`; later it is stopped and the conversation counts as failed. |
| `seed` | 0 | Seeds the jitter. With the conversation id, it fixes the stream: every submission gets the same one. |

These settings decide when words reach an extractor and how they are grouped. They don't enter any latency or
live score: every clock starts when the words arrive ([metrics.md](metrics.md), "Whose clock?").
`turn_pause_s` changes how often two speakers share a turn.

## matching

| Key | Meaning | Effect |
|---|---|---|
| `fuzzy_name_threshold` | Similarity (0–100) from which two names count as the same | Applies to text fields whose gold has no digit: hotel, restaurant, place and person names. 85 accepts "acorn guesthouse" for "acorn guest house" and refuses a different name. Identifiers (IDs, plates) always match exactly. Also decides where a name counts as said when the gold is built ([data.md](data.md), "The gold values"). |

## score: the ranking

The score is the harmonic mean of completeness, precision, speed and outcome ([metrics.md](metrics.md)).

| Key | Value | Meaning | Effect |
|---|---|---|---|
| `speed_limit_s.spokenwoz` | 8 | A value shown this many seconds after it was said, or later, earns no speed | Lower = a slow form loses speed sooner. On SpokenWOZ it is also how long after the last value the form is read (`T_END`). A placeholder until it is set on real runs, like the next two. |
| `speed_limit_s.ami` | 80 | The same for AMI | Meeting notes may be slower. |
| `speed_power` | 0.5 | The curve: speed = (1 − delay ÷ limit) ^ power | 0.5 is a square-root curve: close to 1 for short delays, falling steeply near the limit. 1 would be a straight line. |
| `dataset_weights` | 0.5 / 0.5 | The expected mix of typed forms (SpokenWOZ) and free-text forms (AMI) | How a submission's two scores are averaged in the ranking. |

## turn_metrics

| Key | Value | Meaning |
|---|---|---|
| `read_after_s` | 3.0 | Joint goal accuracy and turn slot F1 read the form this long after each caller turn's last words arrived. Only these two orientation numbers use it. |

## latency

| Key | Value | Meaning |
|---|---|---|
| `percentile` | 90 | Which percentile of Final Decision latency is reported next to the median. The median hides a slow tail, the maximum is one outlier. The same percentile is reported for how long a model call takes. |

Latency is the extractor's own: it is counted from the moment the words reach it, not from the moment they
were spoken. It is reported, not ranked: the score's speed term is what counts.

## judge: free text (AMI)

| Key | Meaning |
|---|---|
| `kind` | `nli` (default) or `llm_facts`. |
| `nli_model` | The NLI model: a text supports a statement when "entailment" is its most likely label. |
| `llm_model` | For `llm_facts` only. Runs with MLX, or behind an OpenAI-compatible server when `LFE_JUDGE_ENDPOINT` is set. |
| `interval_s` | Bounds the judge calls: a field may have one text judged per this many seconds of the conversation (30 s: 27 texts in 13.5 minutes). A field that changes no more often has every text judged. One that is rewritten more often is only looked at every `interval_s` seconds and at the end; a text replaced before the next look is not seen. A judged text counts from the moment it was published, so nothing is rounded to this interval. |
| `batch` | Text pairs judged at once. |
| `window_step` | For a text that is too long for the NLI model (it reads 512 tokens per question and would drop the end). Such a text is read in windows of as many sentences or lines as fit; each window starts this share of a window after the one before (0.5 = half a window, so two sentences closer than half a window always share one). The text says a statement when one window does. Smaller = more windows, more judge calls, and more chances of a wrong "yes". `docs/metrics.md`, "What the judge can miss". |

## plots

| Key | Meaning |
|---|---|
| `mode` | `light` or `dark`: the colours of the plots, of `report.html` and of the page of `lfe ui`. Not part of the fingerprint. |

## ui: the page of `lfe ui`

The page that plays one conversation and shows its form filling ([ui.md](ui.md)). None of these keys decides a
score, and they are not part of the fingerprint: changing them leaves every run as official as it was.

| Key | Value | Meaning | Effect |
|---|---|---|---|
| `host` | 127.0.0.1 | The address the page is served on | 127.0.0.1 is reachable from this machine only. The page starts programs (extractors), so serve it elsewhere only on a network you trust. |
| `port` | 8765 | Its port | `lfe ui --port` overrides it, for example when another program uses this one. |
| `extractor_folders` | `["."]` | Where extractor classes are looked for, with their subfolders. Relative to the folder `lfe ui` is started in | A class is found when it subclasses `Extractor` and has a `run` method. The files are read, not run. |
| `skip_folders` | the package, tests, notebooks, docs, annotations, runs, results | Folder names that are not searched | Hidden folders and virtual environments are never searched. |
| `refresh_s` | 2 | How often the page looks for new extractors and runs | Lower = a new file shows up sooner; each look reads the folder again. |
| `speeds` | 0.5, 1, 2, 5, 10, 30 | The speeds the page offers | Must hold 1 (real time). At another speed, a live run's delays are not what real time would give. |
| `audio_max_speed` | 4 | The recording is played up to this speed | Faster, the page plays without sound: nobody can follow speech beyond a few times its speed. |
| `score_every_s` | 0.5 | How often the score so far is computed: seconds of real time during a live run, seconds of the conversation when a run is reviewed | Lower = finer graphs and marks that follow the form sooner; more work for the judge on free text. |
| `unwatched_stop_s` | 5 | A live run is stopped this long after its page was closed | So that an extractor never keeps calling a model with nobody watching. A reload of the page counts as closing it. |
| `stop_wait_s` | 30 | How long a stopped live run gets for its extractor's `close()` and for writing its run folder; after that its process is ended | A stop ends `run()` at once and then calls `close()`, where an extractor ends what it opened for the conversation (a session, a connection) and reports its usage. Lower = a hanging extractor is ended sooner, but a `close()` that takes longer is cut off and leaves that open. |
