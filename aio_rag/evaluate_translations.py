"""Score translations from an eval run against the AIO references (pure Python, runs on the Mac or the box).

    python evaluate_translations.py chrf  eval/<run>            chrF per arm, per band, paired bootstrap vs baseline
    python evaluate_translations.py judge-prep eval/<run>       blind folders: greek, reference, candidates A/B/C
    python evaluate_translations.py judge-collect eval/<run>    de-anonymise judge scores and aggregate

Expected layout (written by translate.py prepare-eval and the translation subagents):
    eval/<run>/manifest.json, prompts/<arm>/<file>, references/<file>, out/<arm>/<file>
"""
import sys, os, json, random, collections, shutil, re

STAT = random.Random(0)


def chrf(hyp, ref, order=6, beta=2.0):
    hyp, ref = re.sub(r'\s+', '', hyp), re.sub(r'\s+', '', ref)
    P, R = [], []
    for n in range(1, order + 1):
        h = collections.Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        r = collections.Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        m = sum((h & r).values())
        P.append(m / sum(h.values())); R.append(m / sum(r.values()))
    if not P:
        return 0.0
    p, r = sum(P) / len(P), sum(R) / len(R)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def load(run):
    m = json.load(open(f'{run}/manifest.json'))
    out = {}
    for arm in m['arms']:
        for it in m['items']:
            f = f"{run}/out/{arm}/{it['file']}"
            out[(arm, it['id'])] = open(f).read() if os.path.exists(f) else None
    return m, out


def bootstrap(diffs, n=2000):
    means = sorted(sum(STAT.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]


def cmd_chrf(run):
    m, out = load(run)
    refs = {it['id']: open(f"{run}/references/{it['file']}").read() for it in m['items']}
    scores = {arm: {} for arm in m['arms']}
    for (arm, i), hyp in out.items():
        if hyp is not None:
            scores[arm][i] = chrf(hyp, refs[i])
    bands = sorted({it['band'] for it in m['items']}, key=lambda b: int(b.split('-')[0].rstrip('+')))
    band_of = {it['id']: it['band'] for it in m['items']}
    print(f"{'arm':10} {'n':>4} {'chrF':>6}  " + '  '.join(f'{b:>9}' for b in bands))
    for arm, s in scores.items():
        if not s:
            print(f'{arm:10} no outputs yet'); continue
        per = [sum(v for k, v in s.items() if band_of[k] == b) / max(1, sum(band_of[k] == b for k in s)) for b in bands]
        print(f'{arm:10} {len(s):4} {sum(s.values()) / len(s):6.1f}  ' + '  '.join(f'{v:9.1f}' for v in per))
    base = m.get('base_arm', 'baseline')
    if base in scores:
        for arm in scores:
            if arm == base:
                continue
            common = [i for i in scores[arm] if i in scores[base]]
            if len(common) > 1:
                d = [scores[arm][i] - scores[base][i] for i in common]
                lo, hi = bootstrap(d)
                print(f'{arm} - {base}: {sum(d) / len(d):+.1f} chrF (95% CI {lo:+.1f} to {hi:+.1f}, n={len(d)})')
    json.dump({a: s for a, s in scores.items()}, open(f'{run}/chrf.json', 'w'), indent=1)


RUBRIC = """You are judging a blind test of English translations of Ancient Greek inscriptions from Attica.

Each folder holds:
- greek.txt: the Greek text (Leiden conventions: [ ] = editorial restoration, [...] = unrestored gap)
- reference.txt: the published translation from Attic Inscriptions Online (AIO), the gold standard
- A.txt, B.txt (and C.txt if present): candidate translations by anonymous systems; none saw the reference

Score every candidate against the reference, consulting the Greek where the reference is ambiguous or a candidate
may be defensibly different. Score 1-10:
- accuracy: meaning matches the reference clause by clause; penalise mistranslation, invention, wrong numbers
- completeness: whole text translated; penalise omissions, truncation, summarising, loops, missing output
- terminology: names, demotics, offices, institutions, formulae rendered correctly
- conventions: matches AIO's conventions (name forms, set formula renderings, brackets, gaps, layout)
- fluency: readable English
- overall: holistic usefulness to an epigraphist, weighting accuracy and completeness most
Be strict and consistent across folders. Do not try to identify the systems.

Write <folder>/scores.json exactly as:
{"A": {"accuracy": n, "completeness": n, "terminology": n, "conventions": n, "fluency": n, "overall": n,
       "notes": "<=30 words, specific errors"}, "B": {...}, ..., "ranking": ["best", "...", "worst"]}
Read only the files in your folders and this rubric. Do not use the web.
"""


def cmd_judge_prep(run):
    m, out = load(run)
    rng = random.Random(11)
    mapping = {}
    for it in m['items']:
        if any(out[(arm, it['id'])] is None for arm in m['arms']):
            continue
        d = f"{run}/judge/{it['file'][:-4]}"
        os.makedirs(d, exist_ok=True)
        shutil.copy(f"{run}/references/{it['file'][:-4]}.greek.txt", f'{d}/greek.txt')
        shutil.copy(f"{run}/references/{it['file']}", f'{d}/reference.txt')
        arms = list(m['arms']); rng.shuffle(arms)
        mapping[it['id']] = dict(zip('ABC', arms))
        for letter, arm in mapping[it['id']].items():
            open(f'{d}/{letter}.txt', 'w').write(out[(arm, it['id'])])
    json.dump(mapping, open(f'{run}/judge_mapping.json', 'w'), indent=1)
    open(f'{run}/judge_rubric.txt', 'w').write(RUBRIC)
    print(f'{len(mapping)} judge folders in {run}/judge')


def cmd_judge_collect(run):
    m = json.load(open(f'{run}/manifest.json'))
    mapping = json.load(open(f'{run}/judge_mapping.json'))
    crit = ['accuracy', 'completeness', 'terminology', 'conventions', 'fluency', 'overall']
    agg = {a: collections.defaultdict(list) for a in m['arms']}
    wins, per_item = collections.Counter(), {}
    band_of = {it['id']: it['band'] for it in m['items']}
    for i, mp in mapping.items():
        f = f"{run}/judge/{i.replace('/', '__')}/scores.json"
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        for letter, arm in mp.items():
            for c in crit:
                agg[arm][c].append(s[letter][c])
            agg[arm]['band_' + band_of[i]].append(s[letter]['overall'])
        wins[mp[s['ranking'][0]]] += 1
        per_item[i] = {mp[L]: s[L]['overall'] for L in mp}
    print(f"{'arm':10}" + ''.join(f'{c[:8]:>9}' for c in crit) + '   wins')
    for arm, d in agg.items():
        if d['overall']:
            print(f'{arm:10}' + ''.join(f'{sum(d[c]) / len(d[c]):9.2f}' for c in crit) + f'   {wins[arm]}')
    for arm in m['arms']:
        bands = sorted(k for k in agg[arm] if k.startswith('band_'))
        if bands:
            print(arm, ' '.join(f'{b[5:]}:{sum(agg[arm][b]) / len(agg[arm][b]):.2f}' for b in bands))
    base = m.get('base_arm', 'baseline')
    if base in m['arms']:
        for arm in m['arms']:
            if arm != base:
                d = [v[arm] - v[base] for v in per_item.values() if arm in v and base in v]
                if len(d) > 1:
                    lo, hi = bootstrap(d)
                    print(f'{arm} - {base} overall: {sum(d) / len(d):+.2f} (95% CI {lo:+.2f} to {hi:+.2f}, n={len(d)})')
    json.dump(per_item, open(f'{run}/judge_overall.json', 'w'), indent=1)


if __name__ == '__main__':
    {'chrf': cmd_chrf, 'judge-prep': cmd_judge_prep, 'judge-collect': cmd_judge_collect}[sys.argv[1]](sys.argv[2])
