# src/eval_accuracy_plus.py
import argparse
import json
import re
from collections import Counter
from typing import List, Tuple

from rouge_score import rouge_scorer


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
    pt, gt = _tokens(pred), _tokens(gold)
    if not gt:
        return 0
    if len(gt) > short_thresh:
        return 0
    pnorm, gnorm = " ".join(pt), " ".join(gt)
    return int(gnorm in pnorm or pnorm in gnorm)


def f1_score(pred: str, gold: str) -> float:
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


_sc = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)


def rougeL_score(pred: str, gold: str) -> float:
    # F1 Score (0-1)
    return _sc.score(gold, pred)["rougeLsum"].fmeasure


def eval_file(
    pred_file: str,
    short_thresh: int = 15,
    rouge_thresh: float = 0.5
) -> None:
    total = 0
    em_hits = 0
    relaxed_hits = 0
    f1_sum = 0.0
    rsum = 0.0
    r_hits = 0

    # Short-answer subset only
    s_total = 0
    s_em_hits = 0
    s_relaxed_hits = 0
    s_f1_sum = 0.0
    s_rsum = 0.0
    s_r_hits = 0

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            gold = (ex.get("gold") or "").strip()
            pred = (ex.get("prediction") or "").strip()
            total += 1

            em = exact_match(pred, gold)
            em_hits += em

            rel = relaxed_match(pred, gold, short_thresh=short_thresh)
            relaxed_hits += rel

            f1 = f1_score(pred, gold)
            f1_sum += f1

            r = rougeL_score(pred, gold)
            rsum += r
            r_hits += int(r >= rouge_thresh)

            # Short answer subset (based on the number of gold tokens)
            if len(_tokens(gold)) <= short_thresh:
                s_total += 1
                s_em_hits += em
                s_relaxed_hits += rel
                s_f1_sum += f1
                s_rsum += r
                s_r_hits += int(r >= rouge_thresh)

    if total == 0:
        print("No examples to evaluate.")
        return

    em_acc = em_hits / total
    relaxed_acc = max(em_acc, relaxed_hits / total)
    f1_avg = f1_sum / total
    r_avg = rsum / total
    r_acc = r_hits / total

    print(f"Total examples: {total}")
    print(f"Exact Match (accuracy): {em_acc:.4f}")
    print(f"Relaxed Match (≤{short_thresh} tokens): {relaxed_acc:.4f}")
    print(f"Token-level F1 (avg): {f1_avg:.4f}")
    print(f"Rouge-L (avg F): {r_avg:.4f}")
    print(f"Accuracy@Rouge-L>={rouge_thresh}: {r_acc:.4f}")

    if s_total > 0:
        s_em_acc = s_em_hits / s_total
        s_relaxed_acc = max(s_em_acc, s_relaxed_hits / s_total)
        s_f1_avg = s_f1_sum / s_total
        s_r_avg = s_rsum / s_total
        s_r_acc = s_r_hits / s_total
        print("\n-- Short-Answer Subset --")
        print(f"Subset size (gold ≤ {short_thresh} tokens): {s_total}")
        print(f"Exact Match (accuracy): {s_em_acc:.4f}")
        print(f"Relaxed Match: {s_relaxed_acc:.4f}")
        print(f"Token-level F1 (avg): {s_f1_avg:.4f}")
        print(f"Rouge-L (avg F): {s_r_avg:.4f}")
        print(f"Accuracy@Rouge-L>={rouge_thresh}: {s_r_acc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", default="logs/baseline_smol2_test.jsonl")
    ap.add_argument("--short_thresh", type=int, default=15)
    ap.add_argument("--rouge_thresh", type=float, default=0.5)
    args = ap.parse_args()
    eval_file(args.pred_file, args.short_thresh, args.rouge_thresh)


if __name__ == "__main__":
    main()
