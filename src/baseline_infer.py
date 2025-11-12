# src/baseline_infer.py
import os
import json
import argparse
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.model_load import load_chat_model, DEFAULT_CKPT
from src.data_utils import build_messages
from src.prepare_dataset import load_csv_with_standard_view


def generate_predictions(
    model_name: str,
    test_csv: str,
    out_file: str,
    max_prompt_len: int = 1024,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    # 1) Initialize Model and Tokenizer
    tok, model = load_chat_model(
        checkpoint=model_name,
        device=None,            # cuda
        dtype=dtype,
        device_map=None
    )

    # 2) Load the dataset
    ds = load_csv_with_standard_view(test_csv)
    n = len(ds)
    if n == 0:
        raise ValueError(f"No valid rows after standardizing {test_csv}.")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fout:
        for ex in tqdm(ds, total=n):
            instr = ex.get("instruction", "")
            ctx   = ex.get("context", "")
            gold  = ex.get("response", "")

            # 3) Assemble chat messages and apply the model template
            messages = build_messages(instr, ctx if isinstance(ctx, str) and ctx.strip() else None)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 4) Encoding + Generation
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(1e-6, float(temperature)),
                top_p=top_p,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id
            )

            pred = tok.decode(gen[0], skip_special_tokens=True)

            # 5) Write Jsonl
            fout.write(json.dumps(
                {
                    "instruction": instr,
                    "context": ctx if isinstance(ctx, str) else None,
                    "gold": gold,
                    "prediction": pred
                },
                ensure_ascii=False
            ) + "\n")

    print(f"Saved predictions → {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=DEFAULT_CKPT)
    ap.add_argument("--test_csv",  default=os.path.join("data", "test.csv"))
    ap.add_argument("--out_file",  default=os.path.join("logs", "baseline_smol2_test.jsonl"))
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    generate_predictions(
        model_name=args.model_name,
        test_csv=args.test_csv,
        out_file=args.out_file,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
