# Data licences

This repository contains no dataset. The data you are given (`cf-hackathon-bench-data.zip`) is built from four
public sources and carries their licences; using it means accepting them. It holds the conversations, their forms
and their gold: from SpokenWOZ, and from the AMI corpus, QMSum and MISeD (statements of what AMI's meetings say
with short quotes of their transcripts, and the questions QMSum and MISeD ask about them).

| Source | What is used | Licence | From |
|---|---|---|---|
| **SpokenWOZ** (text) | Dialogues with word times, dialogue states, spans, dialogue acts, ontology | **CC BY-NC 4.0: non-commercial use only** | huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text and …/SpokenWOZ-Test-Text-Fixed |
| **AMI Meeting Corpus**, manual annotations 1.6.2 | Words with times, dialogue acts, abstractive summaries and their links, topic segments, speaker roles | CC BY 4.0 | groups.inf.ed.ac.uk/ami |
| **QMSum** | Queries and reference answers on AMI meetings (the "Product" domain) | MIT | github.com/Yale-LILY/QMSum |
| **MISeD** | Questions, reference answers and their attributions on AMI meetings | **No licence file in its repository** (published by Google Research with its paper) | github.com/google-research-datasets/MISeD |

- SpokenWOZ's licence rules out commercial use of the dataset and of anything derived from it, including what is
  built from it here.
- MISeD states no licence. Check with its authors before any use beyond research and evaluation.
- The data holds no audio: runs and evaluations replay the transcripts.

## Recordings

The page of `lfe ui` can play a conversation's recording. It is fetched when somebody turns the sound on for
that conversation, one recording at a time, into `paths.audio` (outside the repository), under the licence of
its dataset:

| Source | What is fetched | Licence | From |
|---|---|---|---|
| **SpokenWOZ** (audio, test split) | The recording of one call (about 3 MB), read out of the archive of test recordings | **CC BY-NC 4.0: non-commercial use only** | huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Audio-Fixed |
| **AMI Meeting Corpus** (audio) | The headset mix of one meeting (about 27 MB) | CC BY 4.0 | groups.inf.ed.ac.uk/ami (AMICorpusMirror) |

## Citations

- SpokenWOZ: Si et al., "SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents", NeurIPS 2023.
- AMI: Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", MLMI 2005.
- QMSum: Zhong et al., "QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization", NAACL 2021.
- MISeD: Golany et al., "Efficient Data Generation for Source-grounded Information-seeking Dialogs: A Use Case for Meeting Transcripts", Findings of EMNLP 2024.
