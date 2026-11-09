"""Shared helpers: Greek normalisation, markdown section parsing, reference parsing."""
import re, unicodedata, json

GR = re.compile(r'[Ͱ-Ͽἀ-῿]')


def n_greek(t):
    return len(GR.findall(t or ''))


def clean(t):
    """Remove editorial sigla but keep the letters editors restored."""
    t = re.sub(r'\*\*Part [^*]*\*\*|\bPart \d+\b|\*\*[^*]{0,40}\*\*', ' ', t)
    t = re.sub(r'[\[\]\(\)\{\}<>⟦⟧⟨⟩|⸢⸣]', '', t)
    t = re.sub(r'\.{2,}|…|-{2,}|―|—|–|\bvac(at)?\.?\b|\bvv?\.?\b', ' ', t)
    t = re.sub(r'[⋮⁝:;,.?!\d"\'’‘]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def norm(t):
    """Search form: no accents/breathings, lowercase, final sigma -> sigma, Greek letters only."""
    t = unicodedata.normalize('NFD', clean(t).replace('·', ' '))
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn').lower()
    t = t.replace('ς', 'σ').replace('ϲ', 'σ')
    t = re.sub(r'[^Ͱ-Ͽἀ-῿ ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def section(text, heads):
    for h in heads:
        m = re.search(r'^#{2,3} ' + re.escape(h) + r'\n(.*?)(?=^#{1,3} |\Z)', text, re.S | re.M)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def front_matter(text):
    return text.split('\n---', 1)[0]


def fm_field(fm, key):
    m = re.search(r'^' + key + r':\s*"?(.*?)"?\s*$', fm, re.M)
    return m.group(1) if m else None


def refs(fm):
    m = re.search(r'^references:(.*?)(?=^\w[\w_]*:)', fm, re.S | re.M)
    block = m.group(1) if m else ''
    return [r.strip() for r in re.findall(r'"([^"]+)"', block)]


def ref_key(r):
    r = r.replace('³', '3').replace('²', '2').replace('¹', '1')
    return re.sub(r'[\s.,]+', '', r).lower()


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    with open(path, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
