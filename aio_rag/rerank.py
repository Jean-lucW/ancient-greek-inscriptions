"""Cross-encoder reranking with bge-reranker-v2-gemma (the reranker used in Zhuang 2025, AIIIP).

For every test query chunk, the first-pass retriever's top 50 index chunks are re-scored by the reranker
(query Greek vs candidate Greek), and the reordered list is saved to data/rerank/<retriever>+gemma_<prompt>.json.
retrieve.py picks these up as extra retrievers, so coverage selection then runs on the reranked order.

Prompts:
  default  = the model card's generic prompt (as the paper would have used it)
  parallel = adapted to the task: is B a parallel to A sharing formulae or phrasing

Usage: python rerank.py data bm25_char4 [hybrid_char4+e5_raw ...]
"""
import sys, os, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import retrieve as R
from common import clean

D, names = sys.argv[1], sys.argv[2:]
os.makedirs(f'{D}/rerank', exist_ok=True)
NAME = 'BAAI/bge-reranker-v2-gemma'
PROMPTS = {
    'default': "Given a query A and a passage B, determine whether the passage contains an answer to the query "
               "by providing a prediction of either 'Yes' or 'No'.",
    'parallel': "Given an Ancient Greek inscription passage A and another Ancient Greek inscription passage B, "
                "determine whether B is a close parallel to A, sharing its formulae, phrasing or document type, "
                "by providing a prediction of either 'Yes' or 'No'.",
}
tok = AutoTokenizer.from_pretrained(NAME)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to('cuda').eval()
YES = tok('Yes', add_special_tokens=False)['input_ids'][0]
SEP = tok('\n', add_special_tokens=False)['input_ids']


def batch_inputs(pairs, prompt, max_length=512):
    p_ids = tok(prompt, add_special_tokens=False)['input_ids']
    items = []
    for q, p in pairs:
        q_ids = tok(f'A: {q}', add_special_tokens=False, max_length=max_length * 3 // 4, truncation=True)['input_ids']
        b_ids = tok(f'B: {p}', add_special_tokens=False, max_length=max_length, truncation=True)['input_ids']
        ids = ([tok.bos_token_id] + q_ids + SEP + b_ids)[:max_length + 1] + SEP + p_ids
        items.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
    return tok.pad(items, padding=True, pad_to_multiple_of=8, return_tensors='pt').to('cuda')


@torch.no_grad()
def score(pairs, prompt, bs=32):
    out = []
    order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][0]) - len(pairs[i][1]))
    res = [0.0] * len(pairs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = batch_inputs([pairs[i] for i in idx], prompt)
        logits = model(**enc, return_dict=True).logits[:, -1, YES].float().cpu().tolist()
        for i, v in zip(idx, logits):
            res[i] = v
    return res


raw = lambda ci: clean(R.chunks[ci]['greek'])
if os.environ.get('SMOKE'):
    t = time.time()
    qi = R.QRY[0]; js = R.retrievers['bm25_char4'](qi)[:8]
    s = score([(raw(qi), raw(R.IDX[j])) for j in js], PROMPTS['parallel'])
    print('smoke ok', [round(v, 2) for v in s], f'{time.time() - t:.1f}s'); sys.exit()
for name in names:
    first = R.retrievers[name]
    cand = {qi: first(qi)[:R.TOPN] for qi in R.QRY}
    pairs, keys = [], []
    for qi, js in cand.items():
        for j in js:
            pairs.append((raw(qi), raw(R.IDX[j]))); keys.append((qi, j))
    for pname, prompt in PROMPTS.items():
        out_f = f'{D}/rerank/{name}+gemma_{pname}.json'
        if os.path.exists(out_f):
            continue
        t = time.time()
        s = score(pairs, prompt)
        by_q = {}
        for (qi, j), v in zip(keys, s):
            by_q.setdefault(qi, []).append((v, j))
        reranked = {str(qi): [j for v, j in sorted(lst, key=lambda x: -x[0])] for qi, lst in by_q.items()}
        json.dump(reranked, open(out_f, 'w'))
        print(f'{name} {pname}: {len(pairs)} pairs in {time.time() - t:.0f}s', flush=True)
