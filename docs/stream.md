# The live stream: how the replay imitates a speech-to-text service

Your extractor doesn't get a finished transcript. It gets what a streaming speech-to-text service sends during a
conversation: **partials** (the current turn so far, replaced every second or so) and **finished turns**.

No dataset contains such a stream. SpokenWOZ and AMI only have the final words with their times. So the stream is
simulated from those word times, and every parameter of the simulation was measured on real streams.

## What was measured

Four real sessions of AssemblyAI's streaming model (u3-rt-pro), recorded through Speechineer's engine with its
production settings: two news monologues (215 s and 116 s) and two two-person dialogues (130 s and 566 s).
The audio time of every word comes from a second, local speech recogniser (Parakeet TDT 0.6B), so that each
word's arrival in the stream can be compared with the moment it was spoken.

| Measured on the real streams | Value | In the simulation |
|---|---|---|
| Time between two partials of a turn | median 1.21–1.40 s (p10 ≈ 1.1, p90 ≈ 1.6) | `partial_every_s` 1.3 ± `partial_jitter_s` 0.2 |
| New words per partial | median 2–4 | follows from the two settings above |
| A word's end → the first partial or turn that shows it | median 0.75 s in dialogues, 1.06 s in monologues; p90 1.6–1.75 s | `word_delay_s` 0.4: on SpokenWOZ the simulation gives median 0.92 s, p90 1.52 s |
| Where a turn ends | only after a sentence end (0 of 1,137 mid-sentence gaps); there, in 50–75% of the cases when 0.2–0.6 s of silence follows, and in 14 of 16 above 0.7 s | after a sentence end + `turn_pause_s` 0.2 (predicts 102 turn ends, the service made 107) |
| Turn length | median 3.9 s, p75 7.0 s, p90 12.7 s | SpokenWOZ: median 3.6 s, p75 6.2 s, p90 10.3 s |
| Finished turn after its last word | median 0.27 s (p90 0.63 s) | `turn_close_s` 0.3 |
| Continuous speech without a pause | a turn every 60 s | `max_turn_s` 60 |
| Two speakers in one turn | frequent: "What are the office hours, Mrs. Phillips? 8:30 to 4:30 with an hour off for lunch." | 26% of SpokenWOZ's turns |

## The rules of the simulation

1. **Turns.** The conversation's words are taken in order. A turn ends after a word that ends a sentence
   (`.`, `?`, `!`) when at least `turn_pause_s` of silence follows; after any silence of `turn_max_silence_s`;
   or when the turn lasts `max_turn_s`. A speaker change alone doesn't end a turn.
2. **Partials.** While a turn has new words, a partial arrives every `partial_every_s` (± `partial_jitter_s`).
   It holds the turn's words that ended at least `word_delay_s` earlier.
3. **Finished turns.** The finished turn arrives `turn_close_s` after its last word, with all its words, before
   the next turn's first partial.
4. **The same for everyone.** The jitter is seeded with the conversation's id: every submission receives exactly
   the same stream for the same conversation.

## What the stream's delay does to your scores

Nothing. A word reaches your extractor a little after it was spoken, as with a real service, but every clock of
the evaluation starts when the word **arrives**: speed and latency measure your extractor, not the
transcription in front of it. The delay only shapes what your extractor has to work with: text that comes in
batches, turns that end at pauses, a current turn that is still growing.

## Speaker labels

Speaker labels follow AssemblyAI's streaming diarization as its documentation describes it:

- Speakers are `A`, `B`, … in order of first speech. On SpokenWOZ the caller always speaks first, so the caller
  is `A`. On AMI, the form's rules say which letter is which role.
- A turn carries the label of its **dominant** speaker, the one with the most audio in it. A turn with two
  speakers still has one label.
- A turn with less than about a second of audio is labelled `PENDING`.
- In a **finished** turn, every word has its own speaker. A run of words under about a second (a "yes.", a
  "mm-hmm") is `PENDING`: too short to attribute.
- In a **partial**, the words carry no speaker: the service labels only final words.

`stream.speaker_labels: false` removes every label and gives the stream of an adapter without diarization.

## What the simulation leaves out

- **Rewriting.** The real service changed earlier words in 7–27% of its partials. The simulated partials only
  grow, because the datasets have one text per word.
- **Recognition errors.** SpokenWOZ's text already is a recogniser's output for the caller (about 6% word errors,
  such as "courage" for Cambridge). AMI's text is a human transcript.
- **Formatting.** The real service writes "5402" and "1-800-555-7890"; the replay keeps the dataset's spelling.
- **Label mistakes and revisions.** The simulated labels are always right, and the service's later corrections
  of its labels are not sent.
- **Overlapping speech.** When two people talk at once, each one's sentence stays together in the text.

A later stage will replace this replay with a real speech-to-text service on the audio; extractors and the
evaluation stay as they are.
