"""Step 6b: dense embeddings of every chunk's Greek with multilingual-e5-large and voyage-4-nano (local, GPU).

Index chunks are embedded as documents, test chunks as queries. Two text variants:
  raw  = editorial sigla removed, accents kept
  norm = accent-stripped lowercase search form
Usage: python embed.py data/chunks.jsonl data/emb
"""
import sys, os, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from common import clean, norm, read_jsonl

dev = 'cuda'

MODELS = [  # key, hf name, pooling, doc prefix, query prefix, max tokens
    ('e5', 'intfloat/multilingual-e5-large', 'mean', 'passage: ', 'query: ', 512),
    ('voyage', 'voyageai/voyage-4-nano', 'mean', 'Represent the document for retrieval: ',
     'Represent the query for retrieving supporting documents: ', 1024),
]


def encode(model, tok, texts, prefix, pool, maxlen, bs=64):
    out = None
    order = np.argsort([-len(t) for t in texts])
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        enc = tok([prefix + texts[i] for i in idx], padding=True, truncation=True,
                  max_length=maxlen, return_tensors='pt').to(dev)
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()
        if pool == 'cls':
            v = h[:, 0]
        else:
            m = enc['attention_mask'].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
        out[idx] = v
    return out


def main(chunks_path, OUT):
    chunks = read_jsonl(chunks_path)
    os.makedirs(OUT, exist_ok=True)
    is_q = np.array([c['split'] == 'test' for c in chunks])
    np.save(f'{OUT}/is_query.npy', is_q)
    variants = {'raw': [clean(c['greek']) for c in chunks], 'norm': [c['greek_norm'] for c in chunks]}
    for key, name, pool, dp, qp, ml in MODELS:
        if all(os.path.exists(f'{OUT}/{key}_{v}.npy') for v in variants):
            continue
        kw = {'trust_remote_code': True, 'attn_implementation': 'sdpa'} if key == 'voyage' else {}
        dtype = torch.bfloat16 if key == 'voyage' else torch.float16
        model = AutoModel.from_pretrained(name, dtype=dtype, **kw).to(dev).eval()
        tok = AutoTokenizer.from_pretrained(name)
        for v, texts in variants.items():
            docs = encode(model, tok, [t for t, q in zip(texts, is_q) if not q], dp, pool, ml)
            qs = encode(model, tok, [t for t, q in zip(texts, is_q) if q], qp, pool, ml)
            M = np.zeros((len(texts), docs.shape[1]), dtype=np.float32)
            M[~is_q] = docs; M[is_q] = qs
            np.save(f'{OUT}/{key}_{v}.npy', M)
            print(key, v, M.shape, flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
