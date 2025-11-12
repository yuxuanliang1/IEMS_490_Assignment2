# src/infer_lora.py
import os, json, argparse, torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.data_utils import build_messages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--adapter_dir", required=True)  # Traning output
    ap.add_argument("--test_csv", default=os.path.join("data","test.csv"))
    ap.add_argument("--out_file",  default=os.path.join("logs","lora_smol2_test.jsonl"))
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)  # Recommend deterministic settings for evaluation
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    df = pd.read_csv(args.test_csv)
    assert "instruction" in [c.lower() for c in df.columns] or "original-instruction" in [c.lower() for c in df.columns] \
        or "new-instruction" in [c.lower() for c in df.columns], "# Test CSV requires an instruction column (raw or derived)."

    with open(args.out_file, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            instr = row.get("instruction", row.get("original-instruction", ""))
            ctx   = row.get("context", row.get("original-context", ""))
            if isinstance(ctx, float): ctx = ""
            gold  = str(row.get("response", row.get("original-response","")))

            messages = build_messages(str(instr), str(ctx) if isinstance(ctx, str) and ctx.strip() else None)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id
                )
            pred = tok.decode(gen[0], skip_special_tokens=True)

            fout.write(json.dumps({
                "instruction": str(instr),
                "context": str(ctx) if isinstance(ctx, str) else None,
                "gold": gold,
                "prediction": pred
            }, ensure_ascii=False) + "\n")

    print(f"Saved predictions → {args.out_file}")

if __name__ == "__main__":
    main()
