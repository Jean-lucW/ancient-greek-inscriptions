"""bge-reranker-v2-gemma scorer, loaded once on import (GPU). Used by rag.py and rerank.py."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = 'BAAI/bge-reranker-v2-gemma'
PROMPTS = {
    'default': "Given a query A and a passage B, determine whether the passage contains an answer to the query "
               "by providing a prediction of either 'Yes' or 'No'.",
    'parallel': "Given an Ancient Greek inscription passage A and another Ancient Greek inscription passage B, "
                "determine whether B is a close parallel to A, sharing its formulae, phrasing or document type, "
                "by providing a prediction of either 'Yes' or 'No'.",
}
tok = AutoTokenizer.from_pretrained(NAME)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to('cuda').eval()
YES = tok('Yes', add_special_tokens=False)['input_ids'][0]
SEP = tok('\n', add_special_tokens=False)['input_ids']


def _inputs(pairs, prompt, max_length=512):
    p_ids = tok(prompt, add_special_tokens=False)['input_ids']
    items = []
    for q, p in pairs:
        q_ids = tok(f'A: {q}', add_special_tokens=False, max_length=max_length * 3 // 4, truncation=True)['input_ids']
        b_ids = tok(f'B: {p}', add_special_tokens=False, max_length=max_length, truncation=True)['input_ids']
        ids = ([tok.bos_token_id] + q_ids + SEP + b_ids)[:max_length + 1] + SEP + p_ids
        items.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
    return tok.pad(items, padding=True, pad_to_multiple_of=8, return_tensors='pt').to('cuda')


@torch.no_grad()
def score(pairs, prompt='parallel', bs=32):
    prompt = PROMPTS.get(prompt, prompt)
    order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][0]) - len(pairs[i][1]))
    res = [0.0] * len(pairs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        logits = model(**_inputs([pairs[i] for i in idx], prompt), return_dict=True).logits[:, -1, YES].float().cpu().tolist()
        for i, v in zip(idx, logits):
            res[i] = v
    return res
