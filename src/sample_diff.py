# src/sample_diff.py
import argparse, json, random
def load(path): return [json.loads(l) for l in open(path, "r", encoding="utf-8")]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--lora_jsonl", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    b = load(args.baseline_jsonl)
    l = load(args.lora_jsonl)
    # Simple alignment based on the instruction (Note: this is currently the test set)
    keyed = {ex["instruction"]: ex for ex in l}
    cand = [ (ex, keyed.get(ex["instruction"])) for ex in b if ex["instruction"] in keyed ]
    random.seed(42)
    picks = random.sample(cand, min(args.k, len(cand)))
    print("# Error Analysis Samples")
    for i,(be,le) in enumerate(picks,1):
        print(f"\n## Case {i}")
        print("### Instruction")
        print(be.get("instruction",""))
        if (be.get("context") or ""):
            print("\n### Context\n", be.get("context",""))
        print("\n### Gold")
        print(be.get("gold",""))
        print("\n### Baseline Prediction")
        print(be.get("prediction",""))
        print("\n### LoRA Prediction")
        print(le.get("prediction","") if le else "<missing>")
if __name__ == "__main__":
    main()
