import os, json, argparse, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.prepare_dataset import load_csv_with_standard_view
from src.data_utils import build_messages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--test_csv", default=os.path.join("data","test.csv"))
    ap.add_argument("--out_file",  default=os.path.join("logs","lora_greedy_newtok.jsonl"))
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(dev)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    ds = load_csv_with_standard_view(args.test_csv)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as fout:
        for ex in tqdm(ds):
            instr, ctx, gold = ex["instruction"], ex.get("context",""), ex["response"]
            messages = build_messages(instr, ctx if isinstance(ctx,str) and ctx.strip() else None)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(dev)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            prompt_len = inputs["input_ids"].shape[1]
            pred = tok.decode(gen[0][prompt_len:], skip_special_tokens=True).strip()
            fout.write(json.dumps({"instruction": instr, "context": ctx, "gold": gold, "prediction": pred}, ensure_ascii=False)+"\n")
    print(f"Saved → {args.out_file}")

if __name__ == "__main__":
    main()
