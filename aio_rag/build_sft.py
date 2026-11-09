"""Build post-editor training data: (corpus tag + Greek + Opus draft) -> published translation.

    python build_sft.py data/pairs.jsonl train/manifest.json train/out train/sft [--max-tokens 2048]

Texts that fit the token budget are used whole. Longer texts are split into passages at Greek line boundaries;
the draft and the target are cut at the same places with the line aligner from chunk.py, and only when both are
laid out line by line (otherwise the text is dropped and counted). 3% of texts go to a validation file.
Writes train.jsonl / val.jsonl ({"messages": [...], "id", "corpus", "part"}) and stats.json.
"""
import sys, os, re, json, random, argparse, collections
from common import read_jsonl, n_greek
from chunk import content_lines, align_lines

TAG = {'aio': 'AIO', 'ircyr': 'IRCyr', 'iaph': 'IAph', 'igcyr_gvcyr': 'IGCyr', 'iospe': 'IOSPE',
       'isicily': 'ISicily', 'iip': 'IIP', 'eagle': 'EAGLE'}
INSTR = ("Revise the draft English translation of this Greek inscription so that it follows the translation "
         "conventions of the corpus named in the style tag. Correct the draft only where the Greek supports the "
         "change; keep everything the Greek says. Output only the revised translation.")


def user_msg(corpus, greek, draft, part=None):
    head = f"[style: {TAG[corpus]}]" + (f" [passage {part[0]} of {part[1]}]" if part else '')
    return f"{head}\n{INSTR}\n\nGreek:\n{greek.strip()}\n\nDraft:\n{draft.strip()}"


def cuts(g_lines, other_lines, starts):
    """Indices into other_lines where each Greek passage starting at line s begins (monotone)."""
    span = align_lines(g_lines, other_lines)
    out, prev = [], 0
    for s in starts:
        c = 0 if s == 0 else max(prev, span[s][0])
        out.append(c); prev = c
    return out


def prop_pieces(text, fracs):
    """Cut text at the sentence end (. ; :) or line break nearest each fraction of its length (for prose translations)."""
    text = text.strip()
    cands = sorted({m.end() for m in re.finditer(r'[.;:·]\s+|\n+', text)})
    pos, prev = [], 0
    for f in fracs:
        ok = [c for c in cands if c > prev and c < len(text)]
        if not ok:
            return None
        c = min(ok, key=lambda c: abs(c - f * len(text)))
        pos.append(c); prev = c
    edges = [0] + pos + [len(text)]
    pieces = [text[a:b].strip() for a, b in zip(edges, edges[1:])]
    return pieces if all(pieces) else None


STOP = set('the a an of and to in on for by with at from as is be was were that this it his their its he they or '
           'which who whom been are not shall has have had so if all any no other such these those also son'.split())


def cwords(s):
    return {w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in STOP and len(w) > 2}


def sentences(text):
    out = [s.strip() for s in re.split(r'(?<=[.;:·])\s+|\n+', text.strip())]
    return [s for s in out if s]


def bridge_pieces(target, draft_pieces):
    """Assign the target's sentences, in order, to the draft passages they share most content words with
    (monotone segmentation by dynamic programming). Used when the target is prose, not line by line."""
    S, K = sentences(target), len(draft_pieces)
    if len(S) < K:
        return None
    P = [cwords(p) for p in draft_pieces]
    sim = [[len(cwords(s) & P[k]) / max(1, len(cwords(s))) for k in range(K)] for s in S]
    pre = [[0.0] * (len(S) + 1) for _ in range(K)]
    for k in range(K):
        for j, row in enumerate(sim):
            pre[k][j + 1] = pre[k][j] + row[k]
    NEG = float('-inf')
    best = [[NEG] * (len(S) + 1) for _ in range(K + 1)]
    back = [[0] * (len(S) + 1) for _ in range(K + 1)]
    best[0][0] = 0.0
    for k in range(1, K + 1):
        for j in range(k, len(S) - (K - k) + 1):
            for i in range(k - 1, j):
                if best[k - 1][i] == NEG:
                    continue
                v = best[k - 1][i] + pre[k - 1][j] - pre[k - 1][i]
                if v > best[k][j]:
                    best[k][j], back[k][j] = v, i
    if best[K][len(S)] == NEG:
        return None
    edges, j = [len(S)], len(S)
    for k in range(K, 0, -1):
        j = back[k][j]; edges.append(j)
    edges = edges[::-1]
    return [' '.join(S[edges[k]:edges[k + 1]]) for k in range(K)]


def side_pieces(g, other_text, starts, fracs):
    """Split a translation into the Greek passages by line alignment when it follows the inscribed lines;
    None when it does not (the caller then bridges via the draft, or cuts proportionally)."""
    o = content_lines(other_text)
    if 0.6 <= len(o) / len(g) <= 1.6:
        c = cuts(g, o, starts)
        pieces = ['\n'.join(o[c[k]:(c[k + 1] if k + 1 < len(c) else len(o))]) for k in range(len(c))]
        return pieces if all(pieces) else None
    return None


def split_text(greek, draft, target, letters_per_part):
    g = content_lines(greek)
    if len(g) < 2:
        return None
    starts, acc = [0], 0
    for i, l in enumerate(g):
        if acc >= letters_per_part and i > starts[-1]:
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
    tp = side_pieces(g, target, starts, fracs) or bridge_pieces(target, dp)
    if not tp or not all(tp):
        return None
    ends = starts[1:] + [len(g)]
    return [('\n'.join(g[s:e]), dp[k], tp[k]) for k, (s, e) in enumerate(zip(starts, ends))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs'); ap.add_argument('manifest'); ap.add_argument('outdir'); ap.add_argument('sftdir')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--tokenizer', default='unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, fix_mistral_regex=True)
    ntok = lambda s: len(tok(s, add_special_tokens=False)['input_ids'])
    pairs = {p['id']: p for p in read_jsonl(a.pairs)}
    man = json.load(open(a.manifest))
    rng = random.Random(2026)
    val_ids = {it['id'] for it in man if rng.random() < 0.03}
    os.makedirs(a.sftdir, exist_ok=True)
    stats = collections.Counter()
    rows = {'train': [], 'val': []}
    for it in man:
        f = f"{a.outdir}/{it['file']}"
        if not os.path.exists(f):
            stats['no_draft'] += 1; continue
        p, draft = pairs[it['id']], open(f).read()
        target = p['english']
        whole = ntok(user_msg(p['corpus'], p['greek'], draft)) + ntok(target) + 40
        if whole <= a.max_tokens:
            exs = [(None, p['greek'], draft, target)]
        else:
            exs = None
            for lpp in (900, 600, 400, 250, 150):
                parts = split_text(p['greek'], draft, target, lpp)
                if parts and all(ntok(user_msg(p['corpus'], g, d, (1, 1))) + ntok(t) + 40 <= a.max_tokens for g, d, t in parts):
                    exs = [((k + 1, len(parts)), g, d, t) for k, (g, d, t) in enumerate(parts)]
                    break
            if exs is None:
                stats['dropped_too_long'] += 1; continue
            stats['split_texts'] += 1
        split = 'val' if it['id'] in val_ids else 'train'
        for part, g, d, t in exs:
            rows[split].append({'id': it['id'], 'corpus': p['corpus'], 'part': part, 'messages': [
                {'role': 'user', 'content': user_msg(p['corpus'], g, d, part)},
                {'role': 'assistant', 'content': t.strip()}]})
        stats[f'texts_{split}'] += 1
        stats[f'corpus_{p["corpus"]}'] += 1
    for split, r in rows.items():
        with open(f'{a.sftdir}/{split}.jsonl', 'w') as fh:
            for x in r:
                fh.write(json.dumps(x, ensure_ascii=False) + '\n')
        stats[f'examples_{split}'] = len(r)
    json.dump(dict(stats), open(f'{a.sftdir}/stats.json', 'w'), indent=1)
    print(json.dumps(dict(stats), indent=1))


if __name__ == '__main__':
    main()
