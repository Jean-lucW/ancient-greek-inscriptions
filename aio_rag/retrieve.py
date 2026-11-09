"""Steps 6-7 + retrieval check: BM25 / E5 / voyage-4-nano / hybrids, with and without coverage reranking.

For each of the 200 held-out test inscriptions:
  1. its Greek is chunked the same way as the index;
  2. each query chunk retrieves its top 50 index chunks with the given retriever;
  3. K examples are selected, either round-robin by rank ('top') or by coverage reranking ('cov'):
     best hit per query chunk first, then greedily the candidate that adds the most not-yet-covered
     Greek (idf-weighted character 4-grams), max 2 chunks per source inscription, near-duplicates skipped.
Retrieval check (no LLM): how much of the AIO reference translation's wording (content bigrams) appears in
the English of the selected examples. Higher = the examples contain more of the phrasing Opus needs.

Usage: python retrieve.py data   (expects data/chunks.jsonl, data/pairs.jsonl, data/emb/*.npy)
Writes data/retrieval_results.tsv and data/selected_<config>.jsonl
"""
import sys, os, re, math, json, collections, random
import numpy as np
from common import read_jsonl, write_jsonl

D = sys.argv[1]
chunks = read_jsonl(f'{D}/chunks.jsonl')
pairs = {p['id']: p for p in read_jsonl(f'{D}/pairs.jsonl')}
IDX = [i for i, c in enumerate(chunks) if c['split'] == 'index']
QRY = [i for i, c in enumerate(chunks) if c['split'] == 'test']
pos_in_idx = {i: k for k, i in enumerate(IDX)}
N = len(IDX)
test_pairs = collections.OrderedDict()
for i in QRY:
    test_pairs.setdefault(chunks[i]['pair_id'], []).append(i)
TOPN = 50


# ---------------- BM25
def chargrams(text, n=4):
    return [g for w in text.split() for p in ['_' + w + '_'] for g in (p[i:i + n] for i in range(max(1, len(p) - n + 1)))]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        df = collections.Counter(); self.tf = []
        for d in docs:
            c = collections.Counter(d); self.tf.append(c); df.update(c.keys())
        avg = sum(len(d) for d in docs) / len(docs)
        self.idf = {w: math.log(1 + (len(docs) - n + .5) / (n + .5)) for w, n in df.items()}
        self.inv = collections.defaultdict(list)
        for j, c in enumerate(self.tf):
            L = len(docs[j])
            for w, f in c.items():
                self.inv[w].append((j, self.idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))))

    def scores(self, q):
        s = np.zeros(N)
        for w in set(q):
            for j, v in self.inv.get(w, ()):
                s[j] += v
        return s


idx_norm = [chunks[i]['greek_norm'] for i in IDX]
bm_words = BM25([t.split() for t in idx_norm])
bm_char = BM25([chargrams(t) for t in idx_norm])
GRAM_IDF = bm_char.idf


def rank_bm25(model, feats):
    def f(qi):
        s = model.scores(feats(chunks[qi]['greek_norm']))
        top = np.argsort(-s)[:TOPN * 4]
        return [int(j) for j in top]
    return f


retrievers = {
    'bm25_words': rank_bm25(bm_words, lambda t: t.split()),
    'bm25_char4': rank_bm25(bm_char, chargrams),
}

# ---------------- dense
for f in sorted(os.listdir(f'{D}/emb')):
    if not f.endswith('.npy') or f == 'is_query.npy' or 'partial' in f:
        continue
    M = np.load(f'{D}/emb/{f}')
    Md, key = M[IDX], f[:-4]
    def make(M=M, Md=Md):
        def r(qi):
            s = Md @ M[qi]
            return [int(j) for j in np.argsort(-s)[:TOPN * 4]]
        return r
    retrievers[key] = make()


def rrf(*rs):
    def r(qi):
        s = collections.defaultdict(float)
        for rr in rs:
            for rank, j in enumerate(rr(qi)[:200]):
                s[j] += 1 / (60 + rank)
        return sorted(s, key=lambda j: -s[j])
    return r


for k in [k for k in list(retrievers) if k.startswith(('e5', 'voyage'))]:
    retrievers[f'hybrid_char4+{k}'] = rrf(retrievers['bm25_char4'], retrievers[k])

# reranked candidate lists written by rerank.py (bge-reranker-v2-gemma), keyed by query chunk index
if os.path.isdir(f'{D}/rerank'):
    for f in sorted(os.listdir(f'{D}/rerank')):
        if f.endswith('.json'):
            cache_ = {int(k): v for k, v in json.load(open(f'{D}/rerank/{f}')).items()}
            retrievers[f[:-5]] = (lambda c: (lambda qi: c[qi]))(cache_)

# ---------------- selection
gram_sets = [set(chargrams(t)) for t in idx_norm]
parent = [chunks[i]['pair_id'] for i in IDX]


def k_for(nq):
    return max(8, min(30, 2 * nq))


def select(qchunks, ranked, K, mode, cap=2):
    chosen, per_parent = [], collections.Counter()
    def ok(j):
        if j in chosen or per_parent[parent[j]] >= cap:
            return False
        return all(len(gram_sets[j] & gram_sets[c]) / max(1, len(gram_sets[j] | gram_sets[c])) < 0.8 for c in chosen)
    def take(j):
        chosen.append(j); per_parent[parent[j]] += 1
    lists = [r[:TOPN] for r in ranked]
    if mode == 'cov':
        for r in lists:                                   # best hit for every query chunk
            for j in r:
                if ok(j):
                    take(j); break
            if len(chosen) >= K:
                break
        uncovered = set().union(*(set(chargrams(chunks[q]['greek_norm'])) for q in qchunks))
        for j in chosen:
            uncovered -= gram_sets[j]
        best_rank = {}
        for r in lists:
            for rank, j in enumerate(r):
                best_rank[j] = min(best_rank.get(j, 99), rank)
        while len(chosen) < K:
            cands = [j for j in best_rank if ok(j)]
            if not cands:
                break
            gain = lambda j: sum(GRAM_IDF.get(g, 0) for g in gram_sets[j] & uncovered)
            j = max(cands, key=lambda j: (gain(j), -best_rank[j]))
            if gain(j) <= 0:
                break
            take(j); uncovered -= gram_sets[j]
    pos = [0] * len(lists)                                # round-robin fill (all of 'top', remainder of 'cov')
    while len(chosen) < K and any(p < len(l) for p, l in zip(pos, lists)):
        for li, l in enumerate(lists):
            while pos[li] < len(l) and not ok(l[pos[li]]):
                pos[li] += 1
            if pos[li] < len(l) and len(chosen) < K:
                take(l[pos[li]]); pos[li] += 1
    return chosen


# ---------------- retrieval check
STOP = set('the a an of and to in on for by with at from as is be was were that this it his their its he they or '
           'which who whom been are not shall has have had so if all any no other such these those'.split())
def bigrams(t):
    w = re.findall(r'[a-z]+', re.sub(r'<[^>]+>', ' ', t.lower()))
    return {(a, b) for a, b in zip(w, w[1:]) if not (a in STOP and b in STOP)}


bands = [(10, 100), (100, 300), (300, 1000), (1000, 10**9)]
band_of = lambda n: next(f'{lo}-{hi if hi < 10**9 else "+"}' for lo, hi in bands if lo <= n < hi)


def evaluate(names=None):
    rows, rng = [], random.Random(0)
    retrievers['random'] = lambda qi: rng.sample(range(N), TOPN * 4)
    for name, r in retrievers.items():
        if names and name not in names:
            continue
        cache = {qi: r(qi) for qi in QRY}
        for mode in ['top', 'cov']:
            rec, prec, per_band, dump = [], [], collections.defaultdict(list), []
            for pid, qch in test_pairs.items():
                K = k_for(len(qch))
                sel = select(qch, [cache[q] for q in qch], K, mode)
                ref = bigrams(pairs[pid]['english'])
                got = set().union(*(bigrams(chunks[IDX[j]]['english']) for j in sel)) if sel else set()
                r_ = len(ref & got) / max(1, len(ref)); p_ = len(ref & got) / max(1, len(got))
                rec.append(r_); prec.append(p_); per_band[band_of(pairs[pid]['n_greek'])].append(r_)
                dump.append({'pair_id': pid, 'K': K, 'examples': [chunks[IDX[j]]['chunk_id'] for j in sel]})
            rows.append([name, mode, np.mean(rec), np.mean(prec)] + [np.mean(per_band[b]) for b in sorted(per_band, key=lambda b: int(b.split('-')[0]))])
            write_jsonl(f'{D}/selected_{name}_{mode}.jsonl', dump)
            print(f'{name:40} {mode}  recall {np.mean(rec):.3f}  precision {np.mean(prec):.3f}', flush=True)
    hdr = ['retriever', 'select', 'recall', 'precision'] + [f'recall_{b}' for b in sorted(per_band, key=lambda b: int(b.split("-")[0]))]
    with open(f'{D}/retrieval_results.tsv', 'w') as fh:
        fh.write('\t'.join(hdr) + '\n')
        for row in sorted(rows, key=lambda x: -x[2]):
            fh.write('\t'.join([row[0], row[1]] + [f'{v:.3f}' for v in row[2:]]) + '\n')
    print('\n' + open(f'{D}/retrieval_results.tsv').read())


if __name__ == '__main__':
    evaluate()
