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
[Agentic Harness Ancient Greek Inscriptions JLW 11.26.pdf](paper/Agentic%20Harness%20Ancient%20Greek%20Inscriptions%20JLW%2011.26.pdf).

## Post-editor model

The QLoRA adapter for Mistral Small 24B Instruct 2501 is hosted on Hugging Face (private, access on request):
[jlw95/aio-postedit-mistral-small-24b-lora](https://huggingface.co/jlw95/aio-postedit-mistral-small-24b-lora).
Download it to the path the scripts expect:

```
hf download jlw95/aio-postedit-mistral-small-24b-lora --local-dir aio_rag/train/postedit_mistral/adapter
```

The adapter config and model card are also in [aio_rag/train/postedit_mistral/adapter](aio_rag/train/postedit_mistral/adapter).

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
  data/                      pairs.jsonl and chunks.jsonl (the retrieval index), test split, AIO-PHI links
  train/manifest.json, train/out/   the 5,439 Opus drafts the post-editor was trained on
greek data/
  inscriptions/              source corpora: AIO, IRCyr, IAph, IGCyr/GVCyr, IOSPE, ISicily, IIP, EAGLE
  phi-ml/phi.zip             PHI Greek texts, used for AIO entries without their own Greek
.claude/skills/              Claude Code skill for interactive translation
paper/                       the paper (PDF)
```

## Usage

Translate a new inscription (prints the prompt with style guide and parallels for the drafting model):

```
cd aio_rag
python translate.py retrieve --file new_inscription.txt --out prompt.txt
```

Rebuild the index from the corpora:

```
python build_pairs.py "../greek data/inscriptions" "../greek data/phi-ml/phi.zip" data/aio_phi_links.jsonl data/aio_phi_sample_check.jsonl data
python chunk.py data/pairs.jsonl data/chunks.jsonl
```

Retrieval needs Python 3 and NumPy. The post-editor needs a CUDA GPU with 16 GB of memory, PyTorch and unsloth.

## Data

The inscriptions, translations and PHI texts belong to their publishers and remain under their own licences:
Attic Inscriptions Online, Inscriptions of Roman Cyrenaica, Inscriptions of Aphrodisias, IGCyr/GVCyr, IOSPE,
I.Sicily, Inscriptions of Israel/Palestine, EAGLE and the Packard Humanities Institute.
