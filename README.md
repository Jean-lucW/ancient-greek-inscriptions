# Ancient Greek Inscriptions: retrieval-augmented translation

Translates Ancient Greek inscriptions into English in the style of
[Attic Inscriptions Online](https://www.atticinscriptions.com/) (AIO). The pipeline has four stages:

1. **Retrieve** parallels for the Greek: BM25 over character 4-grams of the normalised text, with coverage selection,
   from 7,404 published Greek-English inscription pairs.
2. **Draft** a translation with an LLM (Opus 5.5), given the AIO style guide and the parallels.
3. **Post-edit** the draft with a fine-tuned Mistral Small 24B model that moves it towards the corpus's house style.
4. **Check** every change the post-editor made against the Greek, and undo any change the Greek does not support.

## Results

Measured on 200 held-out Attic texts. chrF is measured against the published AIO translation; judge scores (1-10)
come from a blind LLM judge.

| system | chrF | judge overall |
|---|---|---|
| plain instruction | 59.3 | 7.12 |
| + style guide | 63.4 | 7.64 |
| + style guide + parallels | 68.5 | 8.24 |

Post-editing, on a second judged run over the same 200 texts:

| system | chrF | judge overall |
|---|---|---|
| style guide + parallels | 68.5 | 7.72 |
| + post-editor | 72.9 | 7.04 |
| + post-editor + check | 70.0 | 7.85 |

The post-editor alone matches AIO wording more closely but sometimes invents or drops content; the check step keeps a
small, statistically significant gain. Dense embeddings (multilingual-e5-large, voyage-4-nano, voyage-4-large) and the
bge-reranker-v2-gemma reranker added nothing over BM25 with coverage selection. Full details are in
[aio_rag/README.md](aio_rag/README.md).

## Paper

The write-up of the method and results:
[agentic-harness-ancient-greek-inscriptions.pdf](paper/agentic-harness-ancient-greek-inscriptions.pdf).

## Post-editor model

The QLoRA adapter for Mistral Small 24B Instruct 2501 is hosted on Hugging Face:
[jlw95/aio-postedit-mistral-small-24b-lora](https://huggingface.co/jlw95/aio-postedit-mistral-small-24b-lora).
Download it to the path the scripts expect:

```
hf download jlw95/aio-postedit-mistral-small-24b-lora --local-dir aio_rag/train/postedit_mistral/adapter
```

The adapter config and model card are also in [aio_rag/train/postedit_mistral/adapter](aio_rag/train/postedit_mistral/adapter).

## Data

No inscription data is stored in this repository. The texts and translations belong to their publishers and stay
under their own licences; download them from the sources below to rebuild the index.

| corpus | folder | files | Greek-English source | licence |
|---|---|---|---|---|
| Attic Inscriptions Online (AIO) | `aio` | 3,327 | [atticinscriptions.com](https://www.atticinscriptions.com/) REST API | Creative Commons, per AIO |
| Inscriptions of Roman Cyrenaica (IRCyr2020) | `ircyr` | 1,964 | [olvidalo/efes-ng-ircyr](https://github.com/olvidalo/efes-ng-ircyr) | CC BY 2.0 UK |
| I.Sicily | `isicily` | 1,687 | [ISicily/ISicily](https://github.com/ISicily/ISicily) | CC BY 4.0 |
| Inscriptions of Aphrodisias (IAph2007) | `iaph` | 1,441 | [Aphrodisias/IAph_EFES](https://github.com/Aphrodisias/IAph_EFES) | CC BY 4.0 |
| Inscriptions of Greek Cyrenaica / Greek Verse Inscriptions of Cyrenaica (IGCyr2/GVCyr2) | `igcyr_gvcyr` | 950 | [amsacta.unibo.it/id/eprint/7796](https://amsacta.unibo.it/id/eprint/7796) | CC BY-NC-SA 4.0 |
| IOSPE: Ancient Inscriptions of the Northern Black Sea | `iospe` | 691 | [kingsdigitallab/iospe](https://github.com/kingsdigitallab/iospe) | CC BY 2.0 UK |
| Inscriptions of Israel/Palestine (IIP) | `iip` | 468 | [Brown-University-Library/iip-texts](https://github.com/Brown-University-Library/iip-texts) | CC BY-NC 4.0 |
| EAGLE (Europeana datasets 2058810, 2058803, 2058812) | `eagle` | 320 | [Europeana](https://www.europeana.eu/) bulk download | CC BY-SA 3.0, CC BY-NC-SA 3.0 and 4.0 |

Greek text for AIO entries that publish only a translation comes from the PHI-ML dump of the Packard Humanities
Institute's [Searchable Greek Inscriptions](https://inscriptions.packhum.org/), released with Pythia
([sommerschield/ancient-text-restoration](https://github.com/sommerschield/ancient-text-restoration)). AIO entries
are matched to PHI ids through [aio_rag/data/aio_phi_links.jsonl](aio_rag/data/aio_phi_links.jsonl), checked by hand
on a sample in [aio_rag/data/aio_phi_sample_check.jsonl](aio_rag/data/aio_phi_sample_check.jsonl).

After filtering and deduplication by shared references this gives 7,404 Greek-English pairs, of which 200 Attic texts
are held out for testing ([aio_rag/data/test_ids.json](aio_rag/data/test_ids.json)).

Each corpus is converted to one Markdown file per inscription, with YAML front matter (`title`, `id`, `references`,
`not_before`, `not_after`, `place`, `source`, `licence`, `url`) and `## Ancient Greek text` and
`### English translation` sections. Place the files and the PHI dump as follows:

```
greek_data/
  inscriptions/<corpus folder>/*.md
  phi-ml/phi.zip               PHI-ML texts as phi/<id>.txt
```

Kept in the repository: the AIO-PHI links, the test split, and [aio_rag/train/manifest.json](aio_rag/train/manifest.json)
(the ids and parallels behind the 5,439 drafts the post-editor was trained on). Kept local and gitignored: the
`greek_data/` corpora, the built index (`aio_rag/data/pairs.jsonl`, `chunks.jsonl`), the drafts (`aio_rag/train/out/`),
embeddings and evaluation runs.

## Setup

```
pip install -r requirements.txt        # retrieval, index build, evaluation
pip install -r requirements-gpu.txt    # dense retrievers, reranker, post-editor (CUDA GPU, 16 GB)
```

## Usage

Build the index from the corpora (once):

```
cd aio_rag
python build_pairs.py ../greek_data/inscriptions ../greek_data/phi-ml/phi.zip data/aio_phi_links.jsonl data/aio_phi_sample_check.jsonl data
python chunk.py data/pairs.jsonl data/chunks.jsonl
```

Translate a new inscription (prints the prompt with style guide and parallels for the drafting model):

```
python translate.py retrieve --file new_inscription.txt --out prompt.txt
```

Evaluation and post-editor training are described in [aio_rag/README.md](aio_rag/README.md).

## Repository layout

```
aio_rag/                     pipeline code
  build_pairs.py             assemble and deduplicate Greek-English pairs, hold out the 200 test texts
  chunk.py                   clause chunking and Greek-English line alignment
  rag.py, translate.py       retrieval library and command-line entry point (retrieve, prepare-eval, prepare-train)
  retrieve.py                retrieval comparison (no LLM)
  embed.py, embed_voyage_api.py, rerank.py, rerank_model.py   optional dense retrievers and reranker
  build_sft.py               post-editor training data
  train_postedit.py          QLoRA fine-tune (unsloth)
  postedit_infer.py          run the post-editor over drafts
  check_prep.py              set up the check step
  evaluate_translations.py   chrF, blind judge set-up and scoring
  style_guide.md             AIO translation style guide used in every prompt
  data/                      AIO-PHI links, PHI match sample check, test split
  train/manifest.json        ids and parallels of the post-editor training drafts
  train/postedit_mistral/adapter/   adapter config and model card (weights on Hugging Face)
.claude/skills/              Claude Code skill for interactive translation
paper/                       the paper (PDF)
requirements.txt, requirements-gpu.txt
```

## Licence

Code: [MIT](LICENSE). Post-editor adapter: Apache 2.0, as stated on its model card. Data: see [Data](#data).
