---
base_model: unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- en
- grc
tags:
- lora
- qlora
- unsloth
- ancient-greek
- epigraphy
- translation
- post-editing
---

# AIO post-editor: Mistral Small 24B QLoRA adapter

A LoRA adapter for Mistral Small 3 (24B Instruct 2501) that revises a draft English translation of an Ancient Greek
inscription into the house style of a named corpus, in particular Attic Inscriptions Online (AIO).

It is the post-editing stage of the `aio_rag` pipeline:

1. Retrieve parallels for the Greek text (BM25 over character 4-grams, with coverage selection).
2. Draft a translation with a strong LLM, given the AIO style guide and the parallels.
3. **Post-edit the draft with this adapter.**
4. Check every change the post-editor made against the Greek, and undo any change the Greek does not support.

Step 4 is required. On its own the post-editor matches AIO wording more closely but sometimes invents names, dates and
formulae, drops clauses, strips brackets or loops on long passages.

## Input format

One user turn, Mistral chat template:

```
[style: AIO]
Revise the draft English translation of this Greek inscription so that it follows the translation conventions of the corpus named in the style tag. Correct the draft only where the Greek supports the change; keep everything the Greek says. Output only the revised translation.

Greek:
<Greek text, Leiden conventions>

Draft:
<draft English translation>
```

Style tags seen in training: `AIO`, `IRCyr`, `IAph`, `IGCyr`, `IOSPE`, `ISicily`, `IIP`, `EAGLE`.
Texts longer than 2,048 tokens are split into passages at Greek line boundaries and tagged `[passage i of n]`;
`build_sft.py` and `postedit_infer.py` in the repository do this.

## Usage

```python
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit", max_seq_length=2048, load_in_4bit=True)
model.load_adapter("jlw95/aio-postedit-mistral-small-24b-lora")
FastLanguageModel.for_inference(model)
```

Or from the repository: download the adapter to `aio_rag/train/postedit_mistral/adapter` and run
`python postedit_infer.py <run> <src> <dst>`. Decoding is greedy. Fits on a 16 GB GPU in 4-bit.

## Training

- Data: 5,752 training and 181 validation examples (5,284 and 154 texts) from eight corpora with published
  English translations: AIO 1,709 texts, IRCyr 1,503, IAph 686, IGCyr/GVCyr 572, IOSPE 354, ISicily 343, IIP 267,
  EAGLE 4. Input: style tag, Greek and an Opus 5.5 draft made with the retrieval prompt (leave-one-out parallels).
  Target: the published translation. The 200 held-out AIO evaluation texts were excluded.
- QLoRA, rank 16, alpha 16, dropout 0, all attention and MLP projections; loss on the assistant reply only.
- 2 epochs (720 steps), learning rate 1e-4 cosine, 3% warm-up, effective batch 16, max length 2,048 tokens,
  AdamW 8-bit, bf16, seed 2026. Final validation loss 0.83.

## Evaluation

200 held-out AIO texts, translated by Opus 5.5 with the style guide and BM25 parallels (`rag`), then post-edited by
this adapter (`postedit`), then checked change by change against the Greek (`checked`). Blind Opus 5.5 judge, 1-10.

| arm | chrF | judge overall | accuracy | completeness | conventions |
|---|---|---|---|---|---|
| rag | 68.5 | 7.72 | 8.28 | 9.05 | 6.60 |
| postedit | 72.9 | 7.04 | 7.36 | 8.17 | 6.62 |
| checked | 70.0 | 7.85 | 8.29 | 9.06 | 6.88 |

Judge overall vs `rag`: `postedit` -0.68 (95% CI -0.92 to -0.44), `checked` +0.12 (95% CI +0.06 to +0.19).

## Limitations

Do not use the adapter's output unchecked. Without the check step it adds content the Greek does not contain, omits
content, removes editorial brackets and can repeat itself on long passages. It was trained on inscriptions only.
