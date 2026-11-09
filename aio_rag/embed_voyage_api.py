"""Embed every chunk's Greek with voyage-4-large through the MongoDB-hosted Voyage API (ai.mongodb.com).

Key is read from ~/.config/voyage/api_key. Index chunks -> input_type=document, test chunks -> query.
Free tier without a payment method: 3 requests/min and 10K tokens/min, so batches are capped at ~3,000
estimated tokens and spaced 21 s apart; progress is saved after every batch and the run resumes.
Usage: python embed_voyage_api.py data/chunks.jsonl data/emb
"""
import sys, os, json, time, urllib.request, urllib.error
import numpy as np

MODEL, URL = 'voyage-4-large', 'https://ai.mongodb.com/v1/embeddings'


def _key():
    return open(os.path.expanduser('~/.config/voyage/api_key')).read().strip()


def call(batch, input_type):
    body = json.dumps({'input': batch, 'model': MODEL, 'input_type': input_type}).encode()
    for attempt in range(20):
        try:
            req = urllib.request.Request(URL, body, {'Authorization': f'Bearer {_key()}', 'Content-Type': 'application/json'})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            return [x['embedding'] for x in sorted(d['data'], key=lambda x: x['index'])], d['usage']['total_tokens']
        except urllib.error.HTTPError as e:
            print('HTTP', e.code, e.read()[:200], flush=True)
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except Exception as e:
            print('error', e, flush=True)
        time.sleep(30 + 10 * attempt)
    raise RuntimeError('gave up')


def main(chunks_path, out_dir, budget=3000, gap=21.0):
    from common import clean, read_jsonl
    chunks = read_jsonl(chunks_path)
    texts = [clean(c['greek']) or c['greek'].strip() or '-' for c in chunks]   # API rejects empty strings
    is_q = [c['split'] == 'test' for c in chunks]
    partial = f'{out_dir}/voyage4large_raw.partial.npy'
    M = np.load(partial) if os.path.exists(partial) else np.zeros((len(texts), 1024), dtype=np.float32)
    tokens, last = 0, 0.0
    for input_type, want_q in [('document', False), ('query', True)]:
        todo = [i for i in range(len(texts)) if is_q[i] == want_q and not M[i].any()]
        batches, cur, est = [], [], 0
        for i in todo:
            t = len(texts[i]) + 8
            if cur and est + t > budget:
                batches.append(cur); cur, est = [], 0
            cur.append(i); est += t
        if cur:
            batches.append(cur)
        done = 0
        for b, ids in enumerate(batches):
            wait = gap - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            embs, t = call([texts[i] for i in ids], input_type)
            M[ids] = np.array(embs, dtype=np.float32); tokens += t; done += len(ids)
            np.save(partial, M)
            if b % 10 == 0:
                print(f'{input_type} {done}/{len(todo)} chunks, tokens {tokens}, '
                      f'~{(len(batches) - b - 1) * gap / 60:.0f} min left for this pass', flush=True)
    M /= np.linalg.norm(M, axis=1, keepdims=True).clip(1e-9)
    np.save(f'{out_dir}/voyage4large_raw.npy', M)
    if os.path.exists(partial):
        os.remove(partial)
    print('done', M.shape, 'total tokens', tokens)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
