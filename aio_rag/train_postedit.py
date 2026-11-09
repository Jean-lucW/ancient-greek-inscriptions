"""QLoRA fine-tune of Mistral Small 3 (24B) as a post-editor: (style tag + Greek + Opus draft) -> published translation.

    python train_postedit.py train/sft train/postedit_mistral [--max-steps 5]   # smoke test
    python train_postedit.py train/sft train/postedit_mistral                   # full run, resumes from checkpoints

Loss is on the assistant reply only. Checkpoints every ~20 min of work; rerunning resumes.
"""
import os, sys, argparse, glob
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

ap = argparse.ArgumentParser()
ap.add_argument('sftdir'); ap.add_argument('outdir')
ap.add_argument('--model', default='unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit')
ap.add_argument('--max-seq', type=int, default=2048)
ap.add_argument('--epochs', type=float, default=2)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--max-steps', type=int, default=-1)
a = ap.parse_args()

model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_seq, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    use_gradient_checkpointing='unsloth', random_state=2026)

ds = load_dataset('json', data_files={'train': f'{a.sftdir}/train.jsonl', 'val': f'{a.sftdir}/val.jsonl'})
fmt = lambda b: {'text': [tok.apply_chat_template(m, tokenize=False) for m in b['messages']]}
ds = ds.map(fmt, batched=True, remove_columns=ds['train'].column_names)
print(ds['train'][0]['text'][:600])

smoke = a.max_steps > 0
trainer = SFTTrainer(
    model=model, processing_class=tok, train_dataset=ds['train'], eval_dataset=ds['val'],
    args=SFTConfig(
        output_dir=a.outdir, dataset_text_field='text', max_length=a.max_seq, packing=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, per_device_eval_batch_size=1,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type='cosine',
        warmup_ratio=0.03, weight_decay=0.0, optim='adamw_8bit', bf16=True, logging_steps=5,
        eval_strategy='no' if smoke else 'steps', eval_steps=100, save_strategy='no' if smoke else 'steps',
        save_steps=100, save_total_limit=2, report_to='none', seed=2026))
trainer = train_on_responses_only(trainer, instruction_part='[INST]', response_part='[/INST]')

resume = bool(glob.glob(f'{a.outdir}/checkpoint-*')) and not smoke
stats = trainer.train(resume_from_checkpoint=resume)
print(stats)
if not smoke:
    model.save_pretrained(f'{a.outdir}/adapter'); tok.save_pretrained(f'{a.outdir}/adapter')
    print('saved adapter to', f'{a.outdir}/adapter')
