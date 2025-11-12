# src/eval_accuracy.py
import argparse, json, re
from typing import List
from collections import Counter

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def _tokens(s: str) -> List[str]:
    s = _normalize(s)
    return s.split() if s else []

def exact_match(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))

def relaxed_match(pred: str, gold: str, short_thresh: int = 15) -> int:
    # For short reference answers (word count <= threshold), count a match if one of the strings contains the other.
    pt, gt = _tokens(pred), _tokens(gold)
    if not gt:
        return 0
    if len(gt) > short_thresh:
        return 0
    pnorm, gnorm = " ".join(pt), " ".join(gt)
    return int(gnorm in pnorm or pnorm in gnorm)

def f1_score(pred: str, gold: str) -> float:
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt and not gt: return 1.0
    if not pt or not gt:  return 0.0
    common = Counter(pt) & Counter(gt)
    num_overlap = sum(common.values())
    if num_overlap == 0: return 0.0
    precision = num_overlap / len(pt)
    recall    = num_overlap / len(gt)
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", default="logs/baseline_smol2_test.jsonl")
    ap.add_argument("--short_thresh", type=int, default=15)
    args = ap.parse_args()

    total = 0
    em_hits = 0
    relaxed_hits = 0
    f1_sum = 0.0

    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            gold = ex.get("gold", "") or ""
            pred = ex.get("prediction", "") or ""
            total += 1
            em_hits       += exact_match(pred, gold)
            relaxed_hits  += relaxed_match(pred, gold, args.short_thresh)
            f1_sum        += f1_score(pred, gold)

    if total == 0:
        print("No examples to evaluate.")
        return

    em_acc = em_hits / total
    relaxed_acc = max(em_acc, relaxed_hits / total)
    f1_avg = f1_sum / total

    print(f"Total examples: {total}")
    print(f"Exact Match (accuracy): {em_acc:.4f}")
    print(f"Relaxed Match (short≤{args.short_thresh} tokens): {relaxed_acc:.4f}")
    print(f"Token-level F1 (avg): {f1_avg:.4f}")

if __name__ == "__main__":
    main()
