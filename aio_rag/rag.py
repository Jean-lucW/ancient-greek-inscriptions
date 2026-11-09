"""Retrieval library for translating Greek inscriptions with parallels.

    idx = Index('data', include_test=False)            # eval: test inscriptions kept out of the index
    hits = idx.search(greek, retriever='bm25', rerank=False)
    prompt = build_prompt(greek, hits, arm='rag')

Retrievers:
  bm25                BM25 over character 4-grams of the normalised Greek
  bm25+<emb>          BM25 fused (reciprocal rank) with a dense model; <emb> is the file stem in data/emb,
                      e.g. e5_raw, voyage_raw (voyage-4-nano), voyage4large_raw (voyage-4-large API)
Rerank: bge-reranker-v2-gemma re-scores each chunk's top 50 before coverage selection.
"""
import os, math, collections, json
import numpy as np
from common import norm, clean, n_greek, read_jsonl
import chunk as C

TOPN = 50
HERE = os.path.dirname(os.path.abspath(__file__))


def chargrams(text, n=4):
    return [g for w in text.split() for p in ['_' + w + '_'] for g in (p[i:i + n] for i in range(max(1, len(p) - n + 1)))]


def chunk_query(greek):
    """Split a new inscription exactly as index texts were split."""
    if n_greek(greek) < C.SHORT:
        return [greek.strip()]
    toks = C.greek_tokens(C.content_lines(greek))
    return [' '.join(t[0] for t in part) for part in C.merge(C.segments(toks))]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.N = len(docs)
        df = collections.Counter()
        tfs = []
        for d in docs:
            c = collections.Counter(d); tfs.append(c); df.update(c.keys())
        avg = sum(len(d) for d in docs) / len(docs)
        self.idf = {w: math.log(1 + (self.N - n + .5) / (n + .5)) for w, n in df.items()}
        self.inv = collections.defaultdict(list)
        for j, c in enumerate(tfs):
            L = len(docs[j])
            for w, f in c.items():
                self.inv[w].append((j, self.idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))))

    def rank(self, q, n=TOPN * 4):
        s = np.zeros(self.N)
        for w in set(q):
            for j, v in self.inv.get(w, ()):
                s[j] += v
        return [int(j) for j in np.argsort(-s)[:n]]


class Index:
    def __init__(self, data_dir=os.path.join(HERE, 'data'), include_test=False):
        self.D = data_dir
        all_chunks = read_jsonl(f'{data_dir}/chunks.jsonl')
        self.all_chunks = all_chunks
        self.rows = [i for i, c in enumerate(all_chunks) if include_test or c['split'] == 'index']
        self.chunks = [all_chunks[i] for i in self.rows]
        self.norms = [c['greek_norm'] for c in self.chunks]
        self.bm25 = BM25([chargrams(t) for t in self.norms])
        self.grams = [set(chargrams(t)) for t in self.norms]
        self.parent = [c['pair_id'] for c in self.chunks]
        self._dense, self._embedders, self._reranker = {}, {}, None

    # ---------- dense
    def dense_matrix(self, key):
        if key not in self._dense:
            self._dense[key] = np.load(f'{self.D}/emb/{key}.npy')[self.rows]
        return self._dense[key]

    def embed_queries(self, key, texts):
        if key.startswith('voyage4large'):
            from embed_voyage_api import call                       # API, input_type=query
            v, _ = call([clean(t) or t.strip() or '-' for t in texts], 'query')
            v = np.array(v, dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        import torch
        from transformers import AutoModel, AutoTokenizer
        from embed import MODELS, encode
        mkey, variant = key.rsplit('_', 1)
        spec = next(m for m in MODELS if m[0] == mkey)
        if mkey not in self._embedders:
            kw = {'trust_remote_code': True, 'attn_implementation': 'sdpa'} if mkey == 'voyage' else {}
            dtype = torch.bfloat16 if mkey == 'voyage' else torch.float16
            self._embedders[mkey] = (AutoModel.from_pretrained(spec[1], dtype=dtype, **kw).to('cuda').eval(),
                                     AutoTokenizer.from_pretrained(spec[1]))
        model, tok = self._embedders[mkey]
        prep = [clean(t) if variant == 'raw' else norm(t) for t in texts]
        return encode(model, tok, prep, spec[4], spec[2], spec[5])

    # ---------- rerank
    def rerank(self, query, cand, prompt='parallel'):
        if self._reranker is None:
            import rerank_model
            self._reranker = rerank_model
        s = self._reranker.score([(clean(query), clean(self.chunks[j]['greek'])) for j in cand], prompt)
        return [j for _, j in sorted(zip(s, cand), key=lambda x: -x[0])]

    # ---------- search
    def candidates(self, qtexts, retriever='bm25', qvecs=None):
        lists = [self.bm25.rank(chargrams(norm(t))) for t in qtexts]
        if retriever == 'bm25':
            return lists
        key = retriever.split('+', 1)[1]
        M = self.dense_matrix(key)
        if qvecs is None:
            qvecs = self.embed_queries(key, qtexts)
        fused = []
        for bm, qv in zip(lists, qvecs):
            dn = [int(j) for j in np.argsort(-(M @ qv))[:200]]
            s = collections.defaultdict(float)
            for r in (bm[:200], dn):
                for rank, j in enumerate(r):
                    s[j] += 1 / (60 + rank)
            fused.append(sorted(s, key=lambda j: -s[j]))
        return fused

    def select(self, qtexts, lists, K, cap=2):
        chosen, per_parent = [], collections.Counter()
        def ok(j):
            if j in chosen or per_parent[self.parent[j]] >= cap:
                return False
            g = self.grams[j]
            return all(len(g & self.grams[c]) / max(1, len(g | self.grams[c])) < 0.8 for c in chosen)
        def take(j):
            chosen.append(j); per_parent[self.parent[j]] += 1
        lists = [r[:TOPN] for r in lists]
        for r in lists:
            for j in r:
                if ok(j):
                    take(j); break
            if len(chosen) >= K:
                break
        uncovered = set().union(*(set(chargrams(norm(t))) for t in qtexts))
        for j in chosen:
            uncovered -= self.grams[j]
        best = {}
        for r in lists:
            for rank, j in enumerate(r):
                best[j] = min(best.get(j, 99), rank)
        idf = self.bm25.idf
        while len(chosen) < K:
            cands = [j for j in best if ok(j)]
            if not cands:
                break
            gain = lambda j: sum(idf.get(g, 0) for g in self.grams[j] & uncovered)
            j = max(cands, key=lambda j: (gain(j), -best[j]))
            if gain(j) <= 0:
                break
            take(j); uncovered -= self.grams[j]
        pos = [0] * len(lists)
        while len(chosen) < K and any(p < len(l) for p, l in zip(pos, lists)):
            for li, l in enumerate(lists):
                while pos[li] < len(l) and not ok(l[pos[li]]):
                    pos[li] += 1
                if pos[li] < len(l) and len(chosen) < K:
                    take(l[pos[li]]); pos[li] += 1
        return chosen

    def parent_grams(self, pid):
        if not hasattr(self, '_pgrams'):
            self._pgrams = collections.defaultdict(set)
            for j, p in enumerate(self.parent):
                self._pgrams[p] |= self.grams[j]
        return self._pgrams[pid]

    def leave_out(self, lists, qtexts, exclude, near_dup=0.9):
        """Drop chunks of the excluded texts and of any text whose Greek is near-identical to the query
        (character 4-gram Jaccard >= near_dup): used to make training drafts without the text's own translation."""
        q = set().union(*(set(chargrams(norm(t))) for t in qtexts))
        bad, ok = set(exclude), set()
        for r in lists:
            for j in r:
                p = self.parent[j]
                if p in bad or p in ok:
                    continue
                g = self.parent_grams(p)
                (bad if len(g & q) / max(1, len(g | q)) >= near_dup else ok).add(p)
        self.last_excluded = sorted(bad - set(exclude))
        return [[j for j in r if self.parent[j] not in bad] for r in lists]

    def search(self, greek, retriever='bm25', rerank=False, K=None, qvecs=None, qtexts=None, exclude=None):
        qtexts = qtexts or chunk_query(greek)
        lists = self.candidates(qtexts, retriever, qvecs)
        if exclude:
            lists = self.leave_out(lists, qtexts, exclude)
        if rerank:
            lists = [self.rerank(q, l[:TOPN]) for q, l in zip(qtexts, lists)]
        K = K or max(8, min(30, 2 * len(qtexts)))
        return [self.chunks[j] for j in self.select(qtexts, lists, K)]


# ---------------- prompts
BASE_INSTRUCTIONS = (
    "The text is an epigraphic edition using Leiden conventions: square brackets [ ] enclose letters restored by "
    "editors, [...] marks a lacuna that has not been restored, and similar editorial sigla may appear. Translate the "
    "whole text, every part of it. Keep restored passages in square brackets in your translation, and show unrestored "
    "gaps as [...]. Keep any section labels such as \"Part 1\". Output only the English translation, with no "
    "commentary or notes.")


def style_guide():
    p = os.path.join(HERE, 'style_guide.md')
    return open(p).read().strip() if os.path.exists(p) else ''


def format_parallels(hits):
    out = []
    for k, h in enumerate(hits, 1):
        where = '' if h['n_chunks'] == 1 else f" (passage {h['idx'] + 1} of {h['n_chunks']}; English alignment approximate)"
        out.append(f"[{k}] {h['pair_id']}{where}\nGreek: {h['greek'].strip()}\nEnglish: {h['english'].strip()}")
    return '\n\n'.join(out)


def build_prompt(greek, hits=None, arm='rag'):
    """arm: baseline (plain instruction), style (style guide only), rag (style guide + parallels)."""
    parts = ["Translate the following Ancient Greek inscription into English."]
    if arm in ('style', 'rag'):
        parts.append("Follow the conventions of Attic Inscriptions Online (AIO), summarised in this style guide:\n\n"
                     + style_guide())
    if arm == 'rag' and hits:
        parts.append("Below are parallel passages from other inscriptions with their published AIO-style English "
                     "translations. Use them for how formulae, titles, names and conventions are rendered. Translate "
                     "only the Greek of the new inscription: do not copy names, numbers or content from the parallels "
                     "that are not in it.\n\n" + format_parallels(hits))
    parts.append(BASE_INSTRUCTIONS)
    parts.append("Greek text:\n\n" + greek.strip())
    return '\n\n---\n\n'.join(parts)
