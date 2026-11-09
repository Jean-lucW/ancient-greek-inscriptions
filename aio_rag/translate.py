"""Translate Greek inscriptions with retrieved parallels.

Interactive (Claude Code runs this, then translates the printed prompt itself):
    python translate.py retrieve --file new.txt [--retriever bm25] [--rerank] [--k 12] [--eval-index]

Evaluation set-up (prompts for the 200 held-out Attic texts; translations are then produced by Opus subagents):
    python translate.py prepare-eval --run pilot --arms baseline,style,rag --limit 20 [--retriever bm25] [--rerank]

Retrievers: bm25 | bm25+e5_raw | bm25+voyage_raw (voyage-4-nano) | bm25+voyage4large_raw (voyage-4-large)
By default `retrieve` searches the full database (index + the 200 test texts); --eval-index excludes the test texts.
"""
import argparse, os, sys, json, random
import numpy as np
from rag import Index, build_prompt, chunk_query
from common import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


def cmd_retrieve(a):
    greek = open(a.file).read() if a.file else sys.stdin.read()
    idx = Index(DATA, include_test=not a.eval_index)
    hits = idx.search(greek, retriever=a.retriever, rerank=a.rerank, K=a.k)
    prompt = build_prompt(greek, hits, arm='rag')
    if a.out:
        open(a.out, 'w').write(prompt)
        print(f'prompt written to {a.out} ({len(hits)} parallels)', file=sys.stderr)
    else:
        print(prompt)
    print('parallels: ' + ', '.join(f"{h['pair_id']}#{h['idx']}" for h in hits), file=sys.stderr)


def band(n):
    return '10-100' if n < 100 else '100-300' if n < 300 else '300-1000' if n < 1000 else '1000+'


def cmd_prepare_eval(a):
    pairs = {p['id']: p for p in read_jsonl(f'{DATA}/pairs.jsonl')}
    test_ids = json.load(open(f'{DATA}/test_ids.json'))
    if a.limit:                                         # stratified pilot: equal numbers per length band
        rng = random.Random(7)
        by_band = {}
        for t in test_ids:
            by_band.setdefault(band(pairs[t]['n_greek']), []).append(t)
        per = a.limit // len(by_band)
        test_ids = [t for b in sorted(by_band) for t in rng.sample(by_band[b], per)]
    arms = a.arms.split(',')
    idx = Index(DATA, include_test=False)
    # test chunks exactly as embedded, so precomputed query vectors line up
    test_chunks = {}
    for i, c in enumerate(idx.all_chunks):
        if c['split'] == 'test':
            test_chunks.setdefault(c['pair_id'], []).append(i)
    qmat = None
    if '+' in a.retriever:
        qmat = np.load(f"{DATA}/emb/{a.retriever.split('+', 1)[1]}.npy")
    run = os.path.join(HERE, 'eval', a.run)
    manifest = {'run': a.run, 'retriever': a.retriever, 'rerank': a.rerank, 'arms': arms, 'items': []}
    for t in test_ids:
        p = pairs[t]
        rows = test_chunks[t]
        qtexts = [idx.all_chunks[i]['greek'] for i in rows]
        hits = []
        if 'rag' in arms:
            qvecs = qmat[rows] if qmat is not None else None
            hits = idx.search(p['greek'], a.retriever, a.rerank, qvecs=qvecs, qtexts=qtexts)
        for arm in arms:
            os.makedirs(f'{run}/prompts/{arm}', exist_ok=True)
            fn = t.replace('/', '__') + '.txt'
            open(f'{run}/prompts/{arm}/{fn}', 'w').write(build_prompt(p['greek'], hits, arm=arm))
        os.makedirs(f'{run}/references', exist_ok=True)
        open(f"{run}/references/{t.replace('/', '__')}.txt", 'w').write(p['english'])
        open(f"{run}/references/{t.replace('/', '__')}.greek.txt", 'w').write(p['greek'])
        manifest['items'].append({'id': t, 'file': t.replace('/', '__') + '.txt', 'n_greek': p['n_greek'],
                                  'band': band(p['n_greek']), 'parallels': [f"{h['pair_id']}#{h['idx']}" for h in hits]})
    json.dump(manifest, open(f'{run}/manifest.json', 'w'), indent=1, ensure_ascii=False)
    sizes = {arm: sum(os.path.getsize(f'{run}/prompts/{arm}/{i["file"]}') for i in manifest['items']) for arm in arms}
    print(f"run {a.run}: {len(test_ids)} texts, arms {arms}, prompt chars {sizes}")


def cmd_prepare_train(a):
    """Prompts for post-editor training drafts: every index text, retrieved exactly as in live use but with the text
    itself (and near-identical texts) left out, from the index without the 200 test texts."""
    pairs = {p['id']: p for p in read_jsonl(f'{DATA}/pairs.jsonl')}
    idx = Index(DATA, include_test=False)
    chunks_of = {}
    for c in idx.chunks:
        chunks_of.setdefault(c['pair_id'], []).append(c['greek'])
    out = os.path.join(HERE, 'train', 'prompts'); os.makedirs(out, exist_ok=True)
    manifest, n_dup = [], 0
    for k, (t, qtexts) in enumerate(chunks_of.items()):
        p = pairs[t]
        hits = idx.search(p['greek'], qtexts=qtexts, exclude={t})
        n_dup += bool(idx.last_excluded)
        fn = t.replace('/', '__') + '.txt'
        open(f'{out}/{fn}', 'w').write(build_prompt(p['greek'], hits, arm='rag'))
        manifest.append({'id': t, 'file': fn, 'corpus': p['corpus'], 'n_greek': p['n_greek'], 'band': band(p['n_greek']),
                         'near_dups_excluded': idx.last_excluded, 'parallels': [f"{h['pair_id']}#{h['idx']}" for h in hits]})
        if k % 500 == 0:
            print(k, len(chunks_of), flush=True)
    json.dump(manifest, open(os.path.join(HERE, 'train', 'manifest.json'), 'w'), indent=1, ensure_ascii=False)
    print(f'{len(manifest)} prompts; {n_dup} texts had near-identical texts excluded')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('retrieve')
    r.add_argument('--file'); r.add_argument('--out')
    r.add_argument('--retriever', default='bm25'); r.add_argument('--rerank', action='store_true')
    r.add_argument('--k', type=int); r.add_argument('--eval-index', action='store_true')
    e = sub.add_parser('prepare-eval')
    e.add_argument('--run', required=True); e.add_argument('--arms', default='baseline,rag')
    e.add_argument('--retriever', default='bm25'); e.add_argument('--rerank', action='store_true')
    e.add_argument('--limit', type=int)
    sub.add_parser('prepare-train')
    a = ap.parse_args()
    {'retrieve': cmd_retrieve, 'prepare-eval': cmd_prepare_eval, 'prepare-train': cmd_prepare_train}[a.cmd](a)
