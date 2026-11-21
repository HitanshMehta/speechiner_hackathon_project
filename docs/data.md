# The data: conversations, forms and gold

The data (`data/processed/` in this repository) holds, for every conversation: its timed words, its form, and its gold
(what each field should show, and from when). It was built from four public sources.

| Dataset | Conversations | Gold from |
|---|---|---|
| **SpokenWOZ** | Phone calls to a booking service in Cambridge. Test: 1,000 calls, 41–452 s (median 159 s). Dev: 500 calls. This repository ships 6 test calls. | The dialogue state after each caller turn, spans, dialogue acts |
| **AMI** | Design-team meetings with four people. 9 meetings of at most 15 minutes (5.1–13.9 min). | The answer key, made from the transcript (below), and AMI's topic segments. The questions come from AMI's summaries, QMSum and MISeD. |

## When was a value said?

Every gold value, item, text and fact has a **said time**: the end of the words that support it, in the audio.
The evaluation then starts each value's clock when the stream delivers those words to the extractor
([metrics.md](metrics.md), "Whose clock?"). Where the said time comes from is recorded per value:

| Source | Meaning |
|---|---|
| `span` | The dataset points at the words: a caller span (SpokenWOZ), a topic segment (AMI). |
| `label` | SpokenWOZ, a label corrected after a check ("Labels checked", below): the end of the words that say it. |
| `rule` | SpokenWOZ, no span: the caller's turn ends with the value's last piece ("… 8 9 5 4."), so it is that turn's end. Also a service that the caller names ("a train"): the end of that word. |
| `llm` | An LLM pointed at the words: a SpokenWOZ value with no span and no rule, or a statement of AMI's answer key. |
| `turn` | The end of the caller's turn: a withdrawn value, a service the caller starts asking about, or a value the LLM found no words for. |
| `end` | Nothing points at words: the end of the conversation (an AMI question that must stay empty). |

## SpokenWOZ

### The form: 37 fields, the same for every call

The dataset stores every slot as a string. The types below come from its ontology (the option lists) and from the
values themselves. About 27–29 fields stay empty in a call, which shows whether an extractor invents values.

| Type | Fields | From |
|---|---|---|
| `select` (14) | hotel: area, price range, type, first day · restaurant: area, price range, food (24 options), day · train: from, to (13 stations), day · attraction: area, type (12) · hospital: department (39) | the ontology's option lists; every gold value of the test split is one of its field's options |
| `checkbox` (2) | hotel: parking, internet | the values are yes / no |
| `integer` (4) | hotel: people, nights · restaurant: people · train: people | whole numbers (1–8 in the test split) |
| `slider` (1) | hotel: stars | whole numbers 0–5, range (0, 5) |
| `time` (5) | restaurant: time · train: leave at, arrive by · taxi: leave at, arrive by | HH:MM |
| `email`, `phone` (2) | caller: email, phone number | the profile slots |
| `text` (8) | names of hotel, restaurant, attraction and caller · taxi: from, to · caller: ID number, plate number | open values; IDs keep their leading zeros |
| `multiselect` (1) | `services`: hotel, restaurant, train, taxi, attraction, hospital, police | **derived**: the services the caller's dialogue acts name (they match the call's goal in 97% of the test calls). A service is said at the word of that turn that names it ("a train", "hotels": 1,158 of the 1,869 in the test split, a median 2.2 s before the turn ends), else at the end of the turn ("a place to stay"). |

### The gold values

The values come from comparing the dialogue state after each caller turn with the state before: 10,022 changes
in the test split, with the checked labels below. 454 of them replace a value ("sunday" → "saturday") and 134
withdraw one; 9 values that their field can't hold are left out (an email of "no").

The dataset labels a value once per caller turn: "60936" becomes "6093608512" in one step. A live form can show
it sooner and in smaller steps, so each change is **cut into the pieces its words carry**. Every piece is one
gold entry: the value as it stands once a word is said, timed at the end of that word. The last piece is the
whole value. The test split has 15,660 gold entries: the 10,022 changes and 5,638 pieces before a whole value.

| Kind of value | A piece is | How it is found | Changes cut (test) |
|---|---|---|---|
| Dictated: phone, email, ID number, plate | what one word carries: a digit ("eight"), several ("double five", "0851"), a letter | The turn's words are turned into the characters such a value is written with, and the value's new characters are looked for as one run. | 1,388 |
| … with recognition errors ("9 6 5 4" for 9354) | the same, with the gold's own characters | The run is aligned by edit distance and accepted when at least half of the characters line up (`atoms.min_aligned`). | 236 |
| Names: the caller's, a hotel's, a restaurant's, an attraction's, a taxi's places | a word of the name | The first stretch of words that the evaluation's own comparison accepts as the name ("stephen walters" for Steven Walters) is where the name is said. It is cut between the name's words. | 320 |
| … only spelled | a letter | followed like a dictated value | 21 |
| … misheard ("thomas galaxy" for Thomas Gillespie) | a word of the name | No stretch counts as the name, so its words are given to the heard words of the dataset's span in order: each to the heard word or words it resembles most, complete at the last of them. Only when the heard words resemble the name (a mean similarity of at least `atoms.min_aligned` between the pairs). | 212 |
| Times | a reading: "1:30" is 01:30, then "pm" makes it 13:30 | The plain readings of the span's words (an hour, minutes, am or pm); the last reading must be the gold value. | 167 |
| Options, yes or no, counts, stars | the whole value | Nothing can be shown before its last word. | – |

- **The whole value comes at the first word where it is complete.** That can be before the end of the dataset's
  span, which often goes on: a name said and then spelled, "nine thirty a m". 76 more names are one piece and
  are placed at their first mention in this way.
- **A change that cannot be cut stays one entry**, at the end of its last word. Most of these have nothing to
  cut: the value is one word, the caller only says "yes" to a proposal or points at it ("the first one"), or
  its words reach the extractor in one message. What is left are values said over several messages whose words
  cannot be followed: 39 of the 1,773 dictated values of the test split ("the bacteria will be able to" for a
  plate), and 220 of the 1,195 names, mostly spans that don't say the name and a few noisy spellings. A time
  has no step between its words unless they read as a time ("half past four" has none).
- **The evaluation keeps one step per stream message.** An extractor sees whole messages, so the pieces that
  arrive with the same message count as one step, the last of them ([metrics.md](metrics.md), "What a field
  should show").

### Their said times (test split)

Where the words of each of the 10,022 changes come from:

| Source | Values | How |
|---|---|---|
| `span` | 7,935 (79%) | `span_info` gives inclusive token positions in the turn's text; without the punctuation tokens, a caller's text is exactly its timed words (true for every caller turn), so a span maps to a word's end time. A dictated value is spanned piece by piece; a confirmation is spanned on its "yes". |
| `label` | 6 | A label corrected after a check: the end of the words that say it ("Labels checked", below). |
| `rule` | 1,199 (12%) | The caller's turn ends with the value's last piece. On values that have a span, this rule lands within 0.3 s of the span in 96–98% of the cases. |
| `llm` | 514 (5%) | An LLM quoted the caller's words. |
| `turn` | 368 (4%) | The end of the caller's turn: 134 withdrawn values, and 234 values that the caller's last turn doesn't say. |

Why the words matter and not just the turn: a value's span ends before its turn does by 2.0 s at the 75th
percentile and 4.1 s at the 90th. Speed is counted in seconds, so "the end of the turn" is not precise enough.

### Labels checked

The dataset's values come from the dialogue acts its annotators marked, and a few are wrong or missing: a name
the caller says while asking about a place is left out, a time is filed under the wrong field, a detail the agent
gives is recorded as the caller's wish. The labels of the official sample's calls were checked against the
transcripts, field by field, by the form's own rules and field descriptions. Where the call leaves no doubt, the
label was corrected: 8 of the 84 fields that hold a value. A corrected value is said at the end of the words that
say it (source `label`); every other label is the dataset's.

### Speakers

The caller speaks first in every call, so in the stream the caller is speaker `A`. The pause between the two
speakers is short (median 0.33 s; 18% of the changes overlap), so many quick answers share a turn with the
question.

## AMI

### The form: one per meeting

The datasets ask 20–30 questions about each meeting: AMI's four summary headings (abstract, decisions,
problems, actions), 7–8 QMSum queries and 9–19 MISeD questions. A meeting's form holds the ones that have
**exactly one right answer** (below), and the topics.

| Field | Type | What it should hold | When |
|---|---|---|---|
| a kept question (`actions`, `qmsum_5`, `mised_8`, …): 6–14 per meeting | `textarea`, one fact per line | The statements of the meeting that answer it: who said, proposed, asked or agreed to what | Each statement at the end of the words that complete it |
| … of which 0–5 per meeting must stay empty | `textarea` | Nothing: the meeting holds no statement that answers the question | – |
| `topics` | `multiselect`, 23 options | AMI's topic segmentation: each segment has one topic of a fixed list ("costing", "user target group", …) | A topic at the end of its first segment. 8 of the 9 meetings; one has segments without topics and gets no such field. |

The form's rules say which speaker letter is which role, as a meeting tool that knows its participants would,
and what an answer is: facts, one per line, each a full sentence that names the person by their role, only what
was said. Each question's prompt is the dataset's question followed by the format, "Answer with facts, one per
line.", because a model writes its answer where it reads the field's prompt. With the format in the rules alone,
Gemini 2.5 Flash wrote 348 of its 356 answers on one line ("Summarize the discussion about …" asks for a
paragraph). A line is one unit of the scoring, so a line that holds several facts is judged as one statement.

### The answer key

The datasets' own answers are not the gold. Each is one annotator's choice of what matters: scored against
AMI's abstract with this evaluation, QMSum's summary of the same meeting finds 34% of its facts and gets a grade
of 0.23. An extractor cannot guess that choice, and a true statement it writes beyond it would count as
invented. So the gold is made from the transcript, and only for the questions that leave no choice.

The key lists everything the meeting says as single, checkable statements ("The Project Manager asked whether the
remote controls should maybe be waterproof."), each said at the end of the words that complete it. For each
question the datasets ask, it says whether the question has exactly one right answer and, if so, which statements
that answer has to make (**required**) and which it may make (**accepted**). It was made by two independent
readers, and keeps only what both agree on.

What the key holds, for the 9 meetings:

| | Count |
|---|---|
| Statements | 1,213 (865 listed by both readers), every quote found in its transcript line |
| Questions the datasets ask | 240 |
| Kept | 89: 70 with an answer, 19 that must stay empty |
| Left out | 56 open for both readers (every abstract and whole-meeting summary), 14 unclear for both, 47 where the readers differ, 32 that ask yes or no, 2 with no statement both require |
| Required statements | 234 (2 per question at the median): the gold's facts |
| Accepted statements | 560: never owed, and never counted as invented |

How it is used ([metrics.md](metrics.md), "Free text (AMI)"): a required statement is a fact the field has to
show from its said time on. A line of your answer is right when the required and accepted statements of its
question support it; any other line is invented, because it is off the question or the meeting does not say it.

What stays uncertain:

- **The readers agree, they are not checked by a person.** They require the same statements to 79% (the rest
  is only accepted), and their lists of what the meeting says hold about 85% of each other.
- **The judge is literal.** "The team discussed screens" does not follow, for it, from the statements about
  screens. That is why the form asks for facts: who said what.

### What is left out, and other choices

- **Questions with no answer stay in** when both readers find none (19): their gold is empty, and anything
  written there is invented. MISeD's own "unanswerable" flags are not used.
- **Follow-up questions are left out** (25): MISeD flags questions that only make sense after the one before
  ("What did they decide about it?"); they can't be a field on their own.
- **The datasets' reference answers are not used.** AMI's summaries, QMSum and MISeD only supply the questions:
  what an answer should hold, and from when, comes from the key, whose statements are timed by the words they
  quote.
- **One meeting is short on purpose:** ES2005a's recording lasts 16 minutes, but only its first 5 minutes are
  transcribed (in AMI, QMSum and MISeD alike). The replay follows the transcript.
- **Roles are labels, speakers are not names:** AMI's four speakers are A–D by headset; the stream relabels them
  by order of first speech.

## Field types the datasets don't use

`url`, `float`, `date`, `datetime` and `template` are supported by the interface, the comparison and
`lfe check`, but no dataset field has that type: neither dataset has gold for them. (SpokenWOZ says weekdays,
never dates.)

## Licences

SpokenWOZ: CC BY-NC 4.0 (non-commercial). AMI annotations: CC BY 4.0. QMSum: MIT. MISeD: no licence file in its
repository. See `DATA_LICENSES.md`.
