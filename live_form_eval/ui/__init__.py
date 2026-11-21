"""`lfe ui`: a page that plays one conversation and shows its form filling.

Two ways to use it (docs/ui.md):

- a **live run**: an extractor found in the folder gets the conversation as a live stream, at a speed you
  choose; whatever it calls is really called;
- a **review**: the same conversation as an earlier run filled it, replayed from the run folder; nothing is
  called.

In both, the page shows the transcript as the extractor receives it, the form with what each field shows and
should show, the score as if the conversation ended at that moment, the price so far, and it can play the
recording along.

    catalog.py   what can be chosen: the forms of the config's sample, the extractors in the folder, the runs
    playback.py  one conversation as the page needs it: its stream, its form, its gold over time, a run's values
    scoring.py   the score so far, with the evaluation's own code
    live.py      a live run: the process the extractor runs in, and what it sends to the page
    worker.py    that process
    server.py    the page's server
    static/      the page itself
"""
