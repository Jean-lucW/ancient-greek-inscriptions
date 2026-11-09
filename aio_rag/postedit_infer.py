"""Run the fine-tuned post-editor over a set of drafts.

    python postedit_infer.py eval/postedit rag postedit [--adapter train/postedit_mistral/adapter] [--limit N]

Reads <run>/manifest.json, the Greek from <run>/references/<id>.greek.txt and the drafts from <run>/out/<src>/;
writes revised translations to <run>/out/<dst>/ (one file per text, skipped if it exists, so reruns resume).
Texts too long for the 2,048-token training length are split into passages exactly as in training (build_sft.py).
Greedy decoding, batched by length.
"""
import os, sys, json, time, argparse
from build_sft import user_msg, content_lines, side_pieces, prop_pieces
from common import n_greek

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--adapter', default='train/postedit_mistral/adapter')
ap.add_argument('--corpus', default='aio')
ap.add_argument('--max-seq', type=int, default=2048)
ap.add_argument('--budget', type=int, default=12000, help='max batch size x padded length, in tokens')
ap.add_argument('--max-batch', type=int, default=8)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

import torch
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(a.adapter, max_seq_length=a.max_seq, load_in_4bit=True)
FastLanguageModel.for_inference(model)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ntok = lambda s: len(tok(s, add_special_tokens=False)['input_ids'])


def fits(greek, draft, part):
    return ntok(user_msg(a.corpus, greek, draft, part)) + int(1.3 * ntok(draft)) + 40 <= a.max_seq


def split_input(greek, draft, lpp):
    g = content_lines(greek)
    if len(g) < 2:
        return None
    starts, acc = [0], 0
    for i, l in enumerate(g):
        if acc >= lpp and i > starts[-1]:
            starts.append(i); acc = 0
        acc += n_greek(l)
    if len(starts) == 1:
        return None
    letters = [n_greek(l) for l in g]
    total = max(1, sum(letters))
    fracs = [sum(letters[:s]) / total for s in starts[1:]]
    dp = side_pieces(g, draft, starts, fracs) or prop_pieces(draft, fracs)
    if not dp:
        return None
    ends = starts[1:] + [len(g)]
    return [('\n'.join(g[s:e]), dp[k]) for k, (s, e) in enumerate(zip(starts, ends))]


man = json.load(open(f'{a.run}/manifest.json'))['items']
os.makedirs(f'{a.run}/out/{a.dst}', exist_ok=True)
todo = [it for it in man if not os.path.exists(f"{a.run}/out/{a.dst}/{it['file']}")]
if a.limit:
    todo = todo[:a.limit]

pieces, nparts, log = [], {}, {}
for it in todo:
    greek = open(f"{a.run}/references/{it['file'][:-4]}.greek.txt").read()
    draft = open(f"{a.run}/out/{a.src}/{it['file']}").read()
    if fits(greek, draft, None):
        parts = [(None, greek, draft)]
    else:
        parts = None
        for lpp in (900, 600, 400, 250, 150):
            sp = split_input(greek, draft, lpp)
            if sp and all(fits(g, d, (1, 1)) for g, d in sp):
                parts = [((k + 1, len(sp)), g, d) for k, (g, d) in enumerate(sp)]
                break
        if parts is None:   # cannot split: fall back to the Opus draft unchanged, and record it
            open(f"{a.run}/out/{a.dst}/{it['file']}", 'w').write(draft)
            log[it['id']] = 'unsplittable, draft kept'
            continue
    nparts[it['file']] = len(parts)
    for k, (part, g, d) in enumerate(parts):
        text = tok.apply_chat_template([{'role': 'user', 'content': user_msg(a.corpus, g, d, part)}],
                                       tokenize=False, add_generation_prompt=True)
        pieces.append(dict(file=it['file'], k=k, text=text, n_in=ntok(text), max_new=int(1.5 * ntok(d)) + 64))
print(f'{len(todo)} texts, {len(pieces)} passages, {sum(v > 1 for v in nparts.values())} split texts', flush=True)

pieces.sort(key=lambda p: p['n_in'] + p['max_new'])
batches, cur = [], []
for p in pieces:
    nxt = cur + [p]
    width = max(q['n_in'] + q['max_new'] for q in nxt)
    if cur and (len(nxt) > a.max_batch or len(nxt) * width > a.budget):
        batches.append(cur); cur = [p]
    else:
        cur = nxt
if cur:
    batches.append(cur)

done, t0, ntoks = {}, time.time(), 0


def generate(batch):
    enc = tok([p['text'] for p in batch], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max(p['max_new'] for p in batch), do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
    res = []
    for row in out[:, enc['input_ids'].shape[1]:]:
        ids = [t for t in row.tolist() if t not in (tok.pad_token_id, tok.eos_token_id)]
        res.append((tok.decode(ids, skip_special_tokens=True).strip(), len(ids)))
    return res


def run(batch):
    try:
        return generate(batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            raise
        h = len(batch) // 2
        return run(batch[:h]) + run(batch[h:])


for bi, batch in enumerate(batches):
    for p, (txt, n) in zip(batch, run(batch)):
        ntoks += n
        if n >= p['max_new']:
            log[p['file']] = log.get(p['file'], '') + f' passage {p["k"] + 1} hit max_new;'
        done.setdefault(p['file'], {})[p['k']] = txt
        if len(done[p['file']]) == nparts[p['file']]:
            out = '\n'.join(done[p['file']][k] for k in range(nparts[p['file']]))
            open(f"{a.run}/out/{a.dst}/{p['file']}", 'w').write(out + '\n')
    el = time.time() - t0
    print(f'batch {bi + 1}/{len(batches)} size {len(batch)}  texts done {sum(len(v) == nparts[f] for f, v in done.items())}'
          f'/{len(nparts)}  {ntoks} tokens  {ntoks / el:.1f} tok/s  {el / 60:.1f} min', flush=True)

json.dump(log, open(f'{a.run}/postedit_log_{a.dst}.json', 'w'), indent=1)
print('notes:', json.dumps(log, indent=1))
