"""Claude check step: folders and batches for Opus agents that vet the post-editor's changes against the Greek.

    python check_prep.py eval/postedit rag postedit checked

Writes <run>/check/<name>/{greek.txt, draft.txt, revised.txt}, <run>/check_instructions.md and
<run>/check_batches.json (lists of folder names, packed to ~110,000 characters or 8 texts per agent).
Agents write <run>/out/<dst>/<file>. The reference translation is never copied into these folders.
"""
import os, sys, json

INSTR = """# Check the post-editor's revisions against the Greek

Each folder holds one Ancient Greek inscription from Attica:
- greek.txt: the Greek text (Leiden conventions: [ ] = editorial restoration, [...] = unrestored gap)
- draft.txt: an English translation made by a strong translator from the Greek with parallels
- revised.txt: the same translation after an automatic post-editor rewrote it into the house style of
  Attic Inscriptions Online (AIO). The post-editor is good at AIO conventions but sometimes invents words,
  names or whole lines that are not in the Greek, drops content, or garbles a passage.

Produce the final translation: the revised text, with every change from the draft checked against the Greek.

- Keep a change when the Greek supports it or it is a matter of style: wording, AIO name forms and formula
  renderings, headings such as **Part 1** or **Column 1**, layout, line breaks, bracket and gap conventions.
- Undo a change when it adds anything the Greek does not contain (names, numbers, offices, clauses, lines),
  removes or shortens content the Greek has, or changes the meaning against the Greek. Restore the draft's
  rendering of that passage, adjusted to the surrounding AIO style.
- Where both versions are wrong against the Greek, correct it from the Greek.
- Do not otherwise retranslate, and do not add notes or commentary.

The AIO style guide is at STYLE_GUIDE; read it once before the first folder.
Output only the final English translation.
"""


def main(run, src, mid, dst):
    man = json.load(open(f'{run}/manifest.json'))['items']
    here = os.path.dirname(os.path.abspath(__file__))
    open(f'{run}/check_instructions.md', 'w').write(INSTR.replace('STYLE_GUIDE', f'{here}/style_guide.md'))
    sizes = []
    for it in man:
        name = it['file'][:-4]
        d = f'{run}/check/{name}'
        os.makedirs(d, exist_ok=True)
        g = open(f"{run}/references/{name}.greek.txt").read()
        dr = open(f"{run}/out/{src}/{it['file']}").read()
        rv = open(f"{run}/out/{mid}/{it['file']}").read()
        for fn, txt in (('greek.txt', g), ('draft.txt', dr), ('revised.txt', rv)):
            open(f'{d}/{fn}', 'w').write(txt)
        sizes.append((name, len(g) + len(dr) + len(rv)))
    sizes.sort(key=lambda x: -x[1])
    batches, cur, n = [], [], 0
    for name, s in sizes:
        if cur and (n + s > 110000 or len(cur) >= 8):
            batches.append(cur); cur, n = [], 0
        cur.append(name); n += s
    if cur:
        batches.append(cur)
    json.dump(batches, open(f'{run}/check_batches.json', 'w'), indent=1)
    os.makedirs(f'{run}/out/{dst}', exist_ok=True)
    print(f'{len(man)} folders, {len(batches)} batches')


if __name__ == '__main__':
    main(*sys.argv[1:5])
