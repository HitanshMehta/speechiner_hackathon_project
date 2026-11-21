# Where it runs: your machine, a GPU, or a free Kaggle notebook

Everything in this repository runs on a laptop without a GPU. One step is faster with one: scoring AMI's notes
and answers, which a language model (the judge) reads. This page says what needs what, and how to use the free
GPU of a Kaggle notebook.

## What needs a GPU, and what doesn't

| Step | What it needs | How long |
|---|---|---|
| Getting the data ([step 3](#3-get-the-data)) | Nothing special. | A few minutes at most |
| `lfe run` | Whatever your extractor calls. The harness itself needs no GPU. | An official run is real time: a few minutes for SpokenWOZ's 6 test calls (4 at a time), about 14 minutes for the AMI meeting. |
| `lfe evaluate` on SpokenWOZ | Nothing: no model is involved. | Seconds |
| `lfe evaluate` on AMI | The judge. It runs on a CUDA GPU if there is one, else on Apple's GPU, else on the CPU. | On a 16 GB MacBook: about a minute per meeting, on the CPU as on the Apple GPU. |

When the judge has to read a text pair it has not read before, `lfe evaluate` loads it and prints where it runs:

    free text is judged by nli:dleemiller/finecat-nli-dbv3-l on cuda:0

A run whose texts were all judged in an earlier evaluation is scored from the kept verdicts: the model is not
loaded and the line is not printed.

**The device changes the speed, not the scores.** The same run evaluated on a CPU and on an Apple GPU gave
identical files, value for value.

## On your own machine

### Without a GPU

Nothing to set up. Evaluating AMI takes a few minutes; on SpokenWOZ you won't notice.

### With an NVIDIA GPU

- **Linux:** `uv sync` installs a PyTorch build with CUDA. The judge finds the GPU by itself.
- **Windows:** the default PyTorch build has no CUDA. Install the CUDA build as [pytorch.org](https://pytorch.org)
  describes under "Get started", into this project's environment.
- **Check:** `uv run python -c "import torch; print(torch.cuda.is_available())"` must print `True`.

### On a Mac

The judge uses Apple's GPU by itself. On a Mac with little free memory the CPU can be faster (it was on ours:
112 s against 134 s for two meetings). `--device` chooses:

```bash
uv run lfe evaluate runs/<run id> --device cpu      # or cuda, mps; the default is auto
```

## On Kaggle, step by step

A Kaggle notebook gives you a GPU for free: at least 30 hours a week (your account page shows your quota), in
sessions of up to 12 hours.

### 1. Create a notebook with a GPU and internet

1. On [kaggle.com](https://www.kaggle.com): **Create → New Notebook**.
2. In the panel on the right, under **Session options**: set **Accelerator** to a GPU (T4 x2 or P100) and
   switch **Internet** on. Kaggle asks you to verify your account with a phone number for both, once.

### 2. Install the environment

Each block below is one notebook cell.

```python
!pip install -q uv
!git clone <this repo's URL> /kaggle/working/live-form-eval
%cd /kaggle/working/live-form-eval
!uv sync
```

`uv sync` brings its own Python 3.12 and every dependency, PyTorch with CUDA included. It takes a few minutes.
If the repository is private, upload it as a Kaggle dataset instead and copy it:
`!cp -r /kaggle/input/<dataset name> /kaggle/working/live-form-eval`.

```python
!uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This must print `True` and the GPU's name. If it prints `False`, the accelerator is not switched on (step 1).

### 3. Get the data

Nothing to do: the data comes with the repository (`data/processed/`), so step 2 brought it along.

### 4. Evaluate a run

**A run you made on your own machine.** Zip its folder (`runs/<run id>/`, with `run.json`, `predictions.jsonl`
and `calls.jsonl`) and add it to the notebook: **Input → Upload → New Dataset**. Kaggle unpacks it under
`/kaggle/input/<dataset name>/`.

```python
!uv run lfe evaluate /kaggle/input/<dataset name>/<run id>
```

**Or run your extractor on Kaggle too.** Put your extractor's file in `/kaggle/working/live-form-eval` (upload it
as a dataset and copy it, or clone your own repository). Give it its API key through **Add-ons → Secrets**:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")   # the name your extractor reads
```

```python
!uv run lfe run my_extractor:MyExtractor --dataset ami --n 2 --speed 5      # a quick development run
!uv run lfe evaluate runs/<run id>
```

An official run is real time, so it takes as long as its conversations. Start it with **Save Version → Save & Run All**:
the notebook then runs in the background, while an open session that sits idle is stopped after a while.

To compare several runs in one report, name them together: `!uv run lfe evaluate runs/<run a> runs/<run b>`.

### 5. Read the results

`lfe evaluate` prints where it wrote: `results/<name>/` in the repository folder, where `<name>` is your
extractor's class name (`results/comparison/` for runs with several names).

```python
import pandas as pd
pd.read_csv("results/<name>/summary.csv").T
```

That folder also holds the report: `report.html` (the score, its four terms and the plots) and, in `forms/`, a
page for every conversation (its timeline, the form at the end next to the gold, every field over time, the
transcript). The report's pages link to each other, so take the folder as a whole:

```python
import shutil
shutil.make_archive("results/<name>", "zip", "results/<name>")     # → results/<name>.zip
```

Everything under `/kaggle/working` is listed in the notebook's **Output** panel: download the zip file from
there, unpack it and open `report.html` in your browser. A form is chosen at the top of the page.

To look at one conversation without leaving the notebook, `notebooks/03_evaluate.ipynb` shows how to draw its
timeline in a cell.

## If something differs

These steps were written from Kaggle's documentation and checked on a laptop (CPU and Apple GPU), not run on
Kaggle itself. If a menu has another name or a cell fails, tell the organisers which step.
