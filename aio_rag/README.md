# aio_rag: retrieval-augmented translation of Greek inscriptions

Code lives here and is mirrored to the GPU box at `~/projects/aio_rag` (`gpu push`); data, embeddings and runs are on the box.

## Pipeline

| step | script | output (box: `~/projects/aio_rag/data`) |
|---|---|---|
| pairs, dedup, 200 Attic test texts | `build_pairs.py` | `pairs.jsonl`, `test_ids.json` |
| clause chunking + alignment | `chunk.py` | `chunks.jsonl` |
| local embeddings (E5, voyage-4-nano) | `embed.py` | `emb/e5_*.npy`, `emb/voyage_*.npy` |
| voyage-4-large (API, free tier) | `embed_voyage_api.py` | `emb/voyage4large_raw.npy` |
| retrieval comparison (no LLM) | `retrieve.py` | `retrieval_results.tsv` |
| bge-reranker-v2-gemma candidates | `rerank.py` | `rerank/*.json` |
| retrieval library | `rag.py`, `rerank_model.py` | |
| AIO style guide | `style_guide.md` | |

## Post-editor model

The fine-tuned post-editor (QLoRA adapter for Mistral Small 24B Instruct 2501) is on Hugging Face:
[jlw95/aio-postedit-mistral-small-24b-lora](https://huggingface.co/jlw95/aio-postedit-mistral-small-24b-lora).
Download it to the path `postedit_infer.py` expects:

```
hf download jlw95/aio-postedit-mistral-small-24b-lora --local-dir train/postedit_mistral/adapter
```

## Interactive use

The `translate-inscription` skill (`.claude/skills/translate-inscription`) runs
`python translate.py retrieve --file <greek>` on the box and Claude translates the printed prompt.

## Evaluation on the 200 held-out Attic texts (Opus subagents, no API key)

1. Prepare prompts (on the box), choosing the retriever that won the retrieval comparison:
   ```
   python translate.py prepare-eval --run full --arms baseline,style,rag --retriever bm25 [--rerank]
   ```
   Arms: `baseline` = plain instruction; `style` = style guide only; `rag` = style guide + parallels.
   `--limit 20` makes a stratified pilot (5 per length band).
2. Pull `eval/<run>/prompts` to the Mac. References stay in `eval/<run>/references` and must never be shown to translators.
3. Translate with Opus subagents (`model: opus`), a few prompts per agent: each agent reads only its prompt files
   and writes the translation to `eval/<run>/out/<arm>/<same file name>`.
4. `python evaluate_translations.py chrf eval/<run>`: chrF per arm and length band, paired bootstrap vs baseline.
5. `python evaluate_translations.py judge-prep eval/<run>`: blind folders (Greek, AIO reference, shuffled A/B/C).
   Opus judge subagents follow `eval/<run>/judge_rubric.txt` and write `scores.json` per folder.
6. `python evaluate_translations.py judge-collect eval/<run>`: mean scores, wins, per-band results, bootstrap vs baseline.

## Results: full run (200 held-out Attic texts, 7 Nov 2026)

Retriever `bm25` (character 4-grams) with coverage selection, no reranker; translator Opus 5.5 subagents; blind Opus 5.5 judge.

| arm | chrF | judge overall (1-10) | judge wins |
|---|---|---|---|
| baseline | 59.3 | 7.12 | 12 |
| style | 63.4 | 7.64 | 42 |
| rag | 68.5 | 8.24 | 146 |

rag - baseline: +9.2 chrF (95% CI +7.9 to +10.6), +1.13 judge overall (95% CI +0.98 to +1.27).
Retrieval check: voyage-4-large, E5, voyage-4-nano and the gemma reranker add nothing over BM25 + coverage, so the default stays `bm25`.

## Post-editor evaluation (200 held-out AIO texts, 2026-11-08)

Arms: `rag` (Opus + style guide + BM25 parallels), `postedit` (rag draft revised by the QLoRA Mistral Small 24B post-editor, `postedit_infer.py`), `checked` (postedit output vetted change by change against the Greek by Opus agents, `check_prep.py`). Run folder `eval/postedit`.

| arm | chrF | judge overall | accuracy | completeness | conventions | overall <= 4 | ranked first |
|---|---|---|---|---|---|---|---|
| rag | 68.5 | 7.72 | 8.28 | 9.05 | 6.60 | 1 | 48 |
| postedit | 72.9 | 7.04 | 7.36 | 8.17 | 6.62 | 14 | 66 |
| checked | 70.0 | 7.85 | 8.29 | 9.06 | 6.88 | 0 | 86 |

Judge overall vs rag: postedit -0.68 (95% CI -0.92 to -0.44), checked +0.12 (+0.06 to +0.19; better on 36 texts, worse on 11, tie 153).
The post-editor matches AIO wording more closely (chrF) but invents names, dates and formulae, drops clauses, strips brackets and loops on long passages (10 passages hit the token cap). The Claude check removes those failures and keeps a small, significant gain.
