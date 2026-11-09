"""Pick ~30 varied AIO pairs from the index split (never the test set) for drafting the AIO style guide.
Usage: python styleguide_samples.py data
"""
import sys, re, random
from common import read_jsonl

D = sys.argv[1]
aio = [r for r in read_jsonl(f'{D}/pairs.jsonl') if r['corpus'] == 'aio' and r['split'] == 'index']
rng = random.Random(5)
kinds = ['decree', 'honour', 'dedicat', 'grave|funerary|epitaph|tomb', 'account|payment', 'law', 'epheb',
         'list|catalogue', 'boundary|horos', 'lease', 'inventor', 'letter|emperor', 'statue base',
         'sacred|sacrifice|cult', 'treaty|alliance']
picked = []
for k in kinds:
    pool = [r for r in aio if re.search(k, (r['title'] or '').lower()) and r not in picked and 40 <= r['n_greek'] <= 2500]
    picked += rng.sample(pool, min(2, len(pool)))
picked = picked[:30]
with open(f'{D}/styleguide_samples.txt', 'w') as fh:
    for r in picked:
        fh.write(f"===== {r['id']} | {r['title']} =====\n[GREEK]\n{r['greek'][:2500]}\n\n[AIO ENGLISH]\n{r['english'][:2500]}\n\n")
print(len(picked), 'samples')
