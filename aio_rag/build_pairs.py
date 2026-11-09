"""Step 1-3: assemble Greek-English pairs, deduplicate by shared references, hold out 200 Attic test texts.

Usage: python build_pairs.py <inscriptions_dir> <phi.zip> <aio_phi_links.jsonl> <sample_check.jsonl> <out_dir>
"""
import sys, os, glob, re, zipfile, random, json
from collections import defaultdict, Counter
from common import n_greek, section, front_matter, fm_field, refs, ref_key, read_jsonl, write_jsonl

INS, PHIZIP, LINKS, CHECK, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
MIN_LETTERS = 10

phi = zipfile.ZipFile(PHIZIP)
phi_names = {os.path.basename(n)[:-4] for n in phi.namelist() if n.startswith('phi/') and n.endswith('.txt')}
links = {r['file']: r.get('phi_ids', []) for r in read_jsonl(LINKS)}

rows = []
for f in sorted(glob.glob(INS + '/*/*.md')):
    corpus = f.split('/')[-2]
    t = open(f).read()
    fm = front_matter(t)
    eng = section(t, ['English translation'])
    if not eng:
        continue
    rs = refs(fm)
    phi_ids = re.findall(r'PHI (\d+)', ' '.join(rs))
    greek = section(t, ['Ancient Greek text', 'Greek text'])
    greek_source = 'corpus'
    if corpus == 'aio' and not n_greek(greek):
        ids = [i for i in links.get(os.path.basename(f), []) if i in phi_names]
        if not ids:
            continue
        pid = max(ids, key=int)                      # newest edition = highest PHI id
        greek = phi.read(f'phi/{pid}.txt').decode('utf8', 'ignore')
        greek_source = f'phi:{pid}'
        phi_ids = sorted(set(phi_ids) | set(links[os.path.basename(f)]), key=int)
    if n_greek(greek) < MIN_LETTERS:
        continue
    nb, na = fm_field(fm, 'not_before'), fm_field(fm, 'not_after')
    rows.append({
        'id': f'{corpus}/{os.path.basename(f)[:-3]}', 'corpus': corpus,
        'title': fm_field(fm, 'title'), 'refs': rs, 'phi_ids': phi_ids,
        'not_before': int(nb) if nb and re.fullmatch(r'-?\d+', nb) else None,
        'not_after': int(na) if na and re.fullmatch(r'-?\d+', na) else None,
        'place': fm_field(fm, 'place'), 'greek_source': greek_source,
        'greek': greek.strip(), 'english': eng.strip(), 'n_greek': n_greek(greek),
    })
print('pairs before dedup', len(rows), Counter(r['corpus'] for r in rows))

# --- deduplicate: union-find over shared PHI ids and normalised references
parent = list(range(len(rows)))
size = [1] * len(rows)
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
owner = {}
def keys_of(r):
    return set([f'phi:{p}' for p in r['phi_ids']] + [f'ref:{ref_key(x)}' for x in r['refs']])
freq = Counter(k for r in rows for k in keys_of(r))   # SEG entries can cover whole groups of stones
for i, r in enumerate(rows):
    for k in keys_of(r):
        if freq[k] > 3:
            continue
        if k in owner:
            a, b = find(i), find(owner[k])
            if a != b and size[a] + size[b] <= 4:      # stop chains of pairwise SEG refs merging unrelated stones
                parent[a] = b; size[b] += size[a]
        else:
            owner[k] = i
groups = defaultdict(list)
for i in range(len(rows)):
    groups[find(i)].append(i)

def rank(i):
    r = rows[i]
    return (r['corpus'] == 'aio' and r['greek_source'] == 'corpus', len(r['english']))
pairs = []
for g in groups.values():
    best = max(g, key=rank)
    r = dict(rows[best]); r['duplicates'] = [rows[j]['id'] for j in g if j != best]
    pairs.append(r)
print('pairs after dedup', len(pairs), Counter(r['corpus'] for r in pairs))

# --- test split: 200 Attic (AIO) texts, 50 per length band; skip pairs the sample check flagged
flagged = {'aio/' + r['id'] for r in read_jsonl(CHECK) if r['label'] != 'match'}
bands = [(10, 100), (100, 300), (300, 1000), (1000, 10**9)]
rng = random.Random(2026)
test_ids = []
for lo, hi in bands:
    pool = sorted(r['id'] for r in pairs if r['corpus'] == 'aio' and lo <= r['n_greek'] < hi and r['id'] not in flagged)
    pick = rng.sample(pool, 50)
    test_ids += pick
    print(f'band {lo}-{hi}: pool {len(pool)}, picked 50')
test = set(test_ids)
for r in pairs:
    r['split'] = 'test' if r['id'] in test else 'index'
write_jsonl(f'{OUT}/pairs.jsonl', pairs)
json.dump(test_ids, open(f'{OUT}/test_ids.json', 'w'), indent=1)
print('index', sum(r['split'] == 'index' for r in pairs), 'test', len(test_ids))
