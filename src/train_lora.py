# src/train_lora.py
import os
import argparse
import inspect
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

from src.prepare_dataset import load_csv_with_standard_view
from src.data_utils import build_messages

torch.set_float32_matmul_precision("high")


@dataclass
class ClmDataCollator(DataCollatorForLanguageModeling):
    pass


def make_tokenize_fn(tok: AutoTokenizer, max_len: int):
    def _fn(batch: Dict[str, List[str]]):
        prompts, full_texts = [], []
        for instr, ctx, resp in zip(
            batch["instruction"],
            batch.get("context", [""] * len(batch["instruction"])),
            batch["response"]
        ):
            messages = build_messages(instr, ctx if isinstance(ctx, str) and ctx.strip() else None)
            # Reserve the assistant turn (with empty content) to ensure correct boundary alignment for the label mask
            templ = tok.apply_chat_template(
                messages + [{"role": "assistant", "content": ""}],
                tokenize=False, add_generation_prompt=False
            )
            prompts.append(templ)
            full_texts.append(templ + str(resp))

        tok_full = tok(full_texts, truncation=True, max_length=max_len, return_tensors=None)
        tok_prompt = tok(prompts, truncation=True, max_length=max_len, return_tensors=None)

        input_ids = tok_full["input_ids"]
        labels = []
        for ids, pids in zip(input_ids, tok_prompt["input_ids"]):
            lab = ids.copy()
            p_len = len(pids)
            for i in range(min(p_len, len(lab))):
                lab[i] = -100
            labels.append(lab)

        tok_full["labels"] = labels
        return tok_full
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--train_csv",  default=os.path.join("data", "train.csv"))
    ap.add_argument("--eval_csv",   default=os.path.join("data", "test.csv"))
    ap.add_argument("--output_dir", default=os.path.join("models", "lora-smol2-1k"))
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Data
    ds_train: Dataset = load_csv_with_standard_view(args.train_csv)
    ds_eval:  Dataset = load_csv_with_standard_view(args.eval_csv)
    ds_train = ds_train.filter(lambda ex: ex["instruction"].strip() != "" and ex["response"].strip() != "")
    ds_eval  = ds_eval.filter(lambda ex: ex["instruction"].strip() != "" and ex["response"].strip() != "")

    # 2) tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16
    )
    model.gradient_checkpointing_enable()

    # 3) LoRA
    lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  # Only Attension
    task_type="CAUSAL_LM"
)

    model = get_peft_model(model, lora_cfg)

    # 4) tokenize + collator
    tokenize_fn = make_tokenize_fn(tok, args.max_seq_len)
    tr_tok = ds_train.map(tokenize_fn, batched=True, remove_columns=ds_train.column_names)
    ev_tok = ds_eval.map(tokenize_fn,  batched=True, remove_columns=ds_eval.column_names)
    collator = ClmDataCollator(tokenizer=tok, mlm=False)

    # 5) TrainingArguments
    args_dict = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=max(1, args.bs),
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,

        # Evaluate/Save during training
        evaluation_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,

        logging_steps=args.logging_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        fp16=True,
        bf16=False,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to="none"
    )
    sig = inspect.signature(TrainingArguments)
    filtered = {k: v for k, v in args_dict.items() if k in sig.parameters}
    if ("evaluation_strategy" not in filtered and
        "eval_strategy" not in filtered and
        "do_eval" in sig.parameters):
        filtered["do_eval"] = True

    targs = TrainingArguments(**filtered)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tr_tok,
        eval_dataset=ev_tok,
        data_collator=collator
    )

    trainer.train()
    model.save_pretrained(args.output_dir)  # Save LoRA
    tok.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
