---
name: translate-inscription
description: Translate an Ancient Greek inscription into English in the style of Attic Inscriptions Online (AIO), using retrieved parallels from the user's inscription database (aio_rag on the GPU box). Use whenever the user pastes Greek inscription text and asks for a translation, or asks to translate an inscription "with the RAG", "with parallels" or "AIO style".
---

# Translate an inscription with retrieved parallels

The retrieval database lives on the GPU box in `~/projects/aio_rag` (code mirrored locally in `aio_rag/`).
Run everything through the GPU helper: `G=~/.claude/skills/gpu/scripts/gpu`.

1. Save the Greek exactly as given (keep brackets and line breaks) to a local scratch file, e.g. `<scratchpad>/query.txt`.
2. Push it and retrieve:
   ```
   $G push <scratchpad>/query.txt projects/aio_rag/tmp/query.txt
   $G sh 'cd ~/projects/aio_rag && python translate.py retrieve --file tmp/query.txt --out tmp/prompt.txt'
   $G pull projects/aio_rag/tmp/prompt.txt <scratchpad>/prompt.txt
   ```
   Defaults: retriever `bm25`, full database. Options: `--rerank` (bge-reranker-v2-gemma), `--retriever bm25+voyage4large_raw`, `--k N`.
   Use the configuration that won the latest evaluation if the user or `aio_rag/README.md` names one.
3. Read `prompt.txt` in full. It contains the AIO style guide, the parallels (Greek + published English) and the
   instructions. Translate the inscription yourself following it exactly: whole text, restorations in [ ], gaps as
   [...], AIO name and formula conventions, line by line. Never copy names or numbers from the parallels.
4. Reply with the translation, then one line listing the parallels used (printed by the retrieve command as `parallels: ...`).
