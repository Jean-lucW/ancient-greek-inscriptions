"""Step 4: clause chunking with Greek-English alignment.

Short texts (< SHORT letters) stay whole. Longer Greek is split at punctuation and Attic formula
boundaries, then merged into chunks of MIN..MAX Greek letters. English for each chunk comes from:
  - 'lines': AIO translations follow the inscribed lines, so Greek and English lines are aligned with a
    length-based dynamic programme and the chunk takes the matching English lines (+1 line margin);
  - 'proportional': other corpora translate as running prose, so the chunk takes the same relative
    span of the English with a margin.

Usage: python chunk.py data/pairs.jsonl data/chunks.jsonl
"""
import sys, re, math
from common import norm, n_greek, read_jsonl, write_jsonl

SHORT, MIN, MAX, HARD = 300, 80, 400, 550
PUNCT = re.compile(r'[··.;:⁝⋮]$')
# normalised first words that open a new clause in decrees
STARTERS = {'εδοξεν', 'επειδη', 'δεδοχθαι', 'επαινεσαι', 'στεφανωσαι', 'αναγραψαι', 'εψηφισθαι',
            'αγαθη', 'αγαθηι', 'εισ', 'στησαι', 'ειναι', 'δουναι', 'καλεσαι', 'ορκοσ', 'θεοι', 'επι'}
SPLIT_AFTER = {'ειπεν', 'ειπε', 'επεστατει', 'εγραμματευεν', 'επρυτανευεν'}
HEADER = re.compile(r'^\s*(\*\*.*\*\*|#+ .*)\s*$')


def content_lines(text):
    return [l for l in text.split('\n') if l.strip() and not HEADER.match(l)]


def greek_tokens(lines):
    toks = []                       # (raw token, normalised, line index)
    for li, line in enumerate(lines):
        for raw in line.split():
            toks.append((raw, norm(raw), li))
    return toks


def segments(toks):
    segs, cur = [], []
    for k, (raw, nm, li) in enumerate(toks):
        first = nm.split(' ')[0] if nm else ''
        if cur and first in STARTERS and n_greek(' '.join(t[0] for t in cur)) >= 30:
            if first != 'εισ' or (k + 2 < len(toks) and toks[k + 2][1].startswith('αναγραφ')):
                if first != 'επι' or (k + 2 < len(toks) and toks[k + 2][1].startswith('αρχοντ')):
                    segs.append(cur); cur = []
        cur.append((raw, nm, li))
        if PUNCT.search(raw) or nm in SPLIT_AFTER:
            segs.append(cur); cur = []
    if cur:
        segs.append(cur)
    return segs


def merge(segs):
    chunks, cur = [], []
    letters = lambda s: n_greek(' '.join(t[0] for t in s))
    for s in segs:
        if letters(s) > HARD:                       # long run without punctuation (lists, accounts)
            if cur:
                chunks.append(cur); cur = []
            piece = []
            for t in s:
                piece.append(t)
                if letters(piece) >= 300:
                    chunks.append(piece); piece = []
            if piece:
                cur = piece
            continue
        if cur and letters(cur) >= MIN and letters(cur) + letters(s) > MAX:
            chunks.append(cur); cur = []
        cur = cur + s
    if cur:
        if chunks and letters(cur) < MIN // 2:
            chunks[-1] = chunks[-1] + cur
        else:
            chunks.append(cur)
    return chunks


def align_lines(g_lines, e_lines):
    """Monotone length-based alignment (Gale-Church style). Returns, per Greek line, the English line span."""
    G = [max(1, n_greek(l)) for l in g_lines]
    E = [max(1, len(l)) for l in e_lines]
    c = sum(E) / sum(G)
    n, m = len(G), len(E)
    INF = float('inf')
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    moves = [(1, 1, 0.0), (1, 0, 2.5), (0, 1, 2.5), (2, 1, 0.6), (1, 2, 0.6)]
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] == INF:
                continue
            for di, dj, pen in moves:
                a, b = i + di, j + dj
                if a > n or b > m:
                    continue
                lg, le = sum(G[i:a]), sum(E[j:b])
                cost = pen + (abs(math.log((le + 1) / (c * lg + 1))) if di and dj else 0)
                if D[i][j] + cost < D[a][b]:
                    D[a][b] = D[i][j] + cost; B[a][b] = (i, j)
    span = [None] * n
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = B[i][j]
        for gi in range(pi, i):
            span[gi] = (pj, max(pj, j - 1))
        i, j = pi, pj
    for gi in range(n):                              # Greek lines aligned to nothing: borrow neighbour
        if span[gi] is None:
            span[gi] = span[gi - 1] if gi else (0, 0)
    return span


def english_proportional(eng, frac0, frac1, margin=0.12):
    a = max(0.0, frac0 - margin) * len(eng)
    b = min(1.0, frac1 + margin) * len(eng)
    a = eng.rfind(' ', 0, int(a)) + 1 if a > 0 else 0
    bb = eng.find(' ', int(b))
    b = len(eng) if bb < 0 else bb
    return ('…' if a > 0 else '') + eng[a:b].strip() + ('…' if b < len(eng) else '')


def chunk_pair(p):
    base = {'pair_id': p['id'], 'corpus': p['corpus'], 'split': p['split']}
    if p['n_greek'] < SHORT:
        return [dict(base, idx=0, n_chunks=1, greek=p['greek'], greek_norm=norm(p['greek']),
                     english=p['english'], align='whole', n_greek=p['n_greek'])]
    g_lines = content_lines(p['greek'])
    e_lines = content_lines(p['english'])
    toks = greek_tokens(g_lines)
    parts = merge(segments(toks))
    use_lines = (p['corpus'] == 'aio' and len(e_lines) >= 3 and
                 0.75 <= len(e_lines) / max(1, len(g_lines)) <= 1.33)
    span = align_lines(g_lines, e_lines) if use_lines else None
    total = sum(n_greek(t[0]) for t in toks)
    out, done = [], 0
    for k, part in enumerate(parts):
        greek = ' '.join(t[0] for t in part)
        ng = n_greek(greek)
        if use_lines:
            l0, l1 = part[0][2], part[-1][2]
            e0 = max(0, span[l0][0] - 1); e1 = min(len(e_lines) - 1, span[l1][1] + 1)
            english = ' '.join(l.strip() for l in e_lines[e0:e1 + 1])
            align = 'lines'
        else:
            flat = re.sub(r'\s+', ' ', ' '.join(e_lines))
            english = english_proportional(flat, done / total, (done + ng) / total)
            align = 'proportional'
        done += ng
        out.append(dict(base, idx=k, n_chunks=len(parts), greek=greek, greek_norm=norm(greek),
                        english=english, align=align, n_greek=ng))
    return out


if __name__ == '__main__':
    pairs = read_jsonl(sys.argv[1])
    chunks = []
    for p in pairs:
        for c in chunk_pair(p):
            c['chunk_id'] = f"{c['pair_id']}#{c['idx']}"
            chunks.append(c)
    write_jsonl(sys.argv[2], chunks)
    from collections import Counter
    idx = [c for c in chunks if c['split'] == 'index']
    print('chunks', len(chunks), 'index', len(idx), 'test', len(chunks) - len(idx))
    print('align', Counter(c['align'] for c in idx))
    ls = sorted(c['n_greek'] for c in idx if c['align'] != 'whole')
    print('multi-chunk letters p10/p50/p90', ls[len(ls) // 10], ls[len(ls) // 2], ls[9 * len(ls) // 10])
