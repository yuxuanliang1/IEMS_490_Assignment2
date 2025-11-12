# IEMS_490_Assignment2
Assignment 2 for Northwestern University IEMS_490 Fall course, focusing on LLM and fine tuning.
## Data
I selected the argilla/databricks-dolly-15k-curated-en dataset from Hugging Face. This dataset contains over 10,000 questions and answers. I randomly selected 15 for testing and 1,000 for training, saving them as .csv and .jsonl files.
[Access the dataset](https://huggingface.co/datasets/argilla/databricks-dolly-15k-curated-en).

## Evaluation Script: `src/eval_accuracy_plus.py`
This script evaluates model predictions against gold answers using a mix of **exact**, **fuzzy**, and **overlap-based** metrics. It expects a JSONL file where each line is a JSON object with at least the fields:

- `"gold"`: reference answer string  
- `"prediction"`: model output string  

Before scoring, both strings are normalized (lowercased, extra whitespace collapsed, punctuation removed).

The script reports:

- **Exact Match (accuracy)** – strict string equality after normalization.  
- **Relaxed Match** – for short answers (≤ `--short_thresh` tokens), counts as correct if either prediction is a substring of the gold answer or vice versa.  
- **Token-level F1 (avg)** – F1 score over normalized word tokens, averaged across all examples.  
- **ROUGE-L (avg F)** – average ROUGE-L F1 using `rouge_score`.  
- **Accuracy@ROUGE-L ≥ `--rouge_thresh`** – fraction of examples whose ROUGE-L F1 meets or exceeds the given threshold.  

In addition, the script computes the same metrics on a **short-answer subset**, defined as examples where the gold answer length is ≤ `--short_thresh` tokens.

## Baseline Inference

This script runs a **greedy decoding baseline** on the test set and saves model predictions in JSONL format for later evaluation.

It:

- Loads a chat-style causal LLM (`AutoModelForCausalLM`) and tokenizer from `--model_name`.
- Reads the test data from `--test_csv` using `load_csv_with_standard_view`.
- Builds messages with `build_messages(instruction, context)` and applies the model’s chat template.
- Truncates the prompt to `--max_prompt_len` tokens and generates up to `--max_new_tokens` tokens with **greedy decoding** (`do_sample=False`).
- Writes one JSON object per line to `--out_file` with fields:
  - `instruction`, `context`, `gold`, `prediction`.

## Model

We use **HuggingFaceTB/SmolLM2-1.7B-Instruct** as the base model for both the baseline and all fine-tuning experiments.

Key reasons for this choice:

- **Compact but capable (1.7B parameters)**  
  Small enough to train and run on a single modern GPU, while still strong enough for instruction-following tasks in this course project.

- **Instruction-tuned**  
  The model is already trained to follow natural language instructions, which matches our data format (`instruction` + optional `context` → `response`) and reduces the amount of task-specific engineering required.

---

### Model Training with LoRA

We fine-tune the base model using **LoRA (Low-Rank Adaptation)** to make training feasible on limited hardware:

- **Parameter-efficient**  
  LoRA adds small trainable rank-`r` matrices to selected attention/MLP layers instead of updating all model weights. This significantly reduces memory usage and training time.

- **Frozen base model**  
  The original SmolLM2-1.7B-Instruct weights remain frozen; only the LoRA parameters are updated. This makes it easy to:
  - store checkpoints in a lightweight format,
  - revert to the base model,
  - and compare “before vs after fine-tuning”.

## Results and Error Analysis

We compare the **baseline model** (no fine-tuning) and the **LoRA-fine-tuned model** using `src/eval_accuracy_plus.py` on a small test set of 15 examples.

### Quantitative Summary

**Overall (15 examples)**

| Model              | EM     | Relaxed | F1      | ROUGE-L (F) | Acc@ROUGE-L ≥ 0.5 |
|--------------------|--------|---------|---------|-------------|--------------------|
| Baseline           | 0.0667 | 0.0667  | 0.3063  | 0.2766      | 0.0667             |
| LoRA (v2, greedy)  | 0.0667 | 0.0667  | 0.3215  | 0.2777      | 0.0667             |

**Short-Answer Subset (gold length ≤ 15 tokens, 4 examples)**

| Model              | EM     | Relaxed | F1      | ROUGE-L (F) | Acc@ROUGE-L ≥ 0.5 |
|--------------------|--------|---------|---------|-------------|--------------------|
| Baseline           | 0.0000 | 0.0000  | 0.2267  | 0.2126      | 0.0000             |
| LoRA (v2, greedy)  | 0.0000 | 0.0000  | 0.1845  | 0.1778      | 0.0000             |

**Key points:**

- On the **full test set**, LoRA slightly improves average F1 (0.3063 → 0.3215) and ROUGE-L (0.2766 → 0.2777), but **exact / relaxed accuracy and Acc@ROUGE-L ≥ 0.5 stay identical**.
- On the **short-answer subset**, LoRA is **worse**: both F1 and ROUGE-L drop, and there are still no exact or relaxed matches.

Overall, the fine-tuned model behaves **differently** from the baseline, but **not clearly better** — especially on short, factoid-style questions.

---

### Training Before vs After: What Actually Changed?

From the samples, we can see that LoRA changes **style and content selection**, but not in a uniformly positive way.

#### 1. Longer, more “chatty” answers → hurts short factual questions

- **Case 3 (RELX stock indexes)**  
  Gold requires a **short, precise list** of indices.  
  - Baseline already adds one extra incorrect item (“New York Stock Exchange”), but still stays relatively short.  
  - LoRA repeats items and adds even more spurious entries (treating stock exchanges themselves as “indexes”), making the answer **longer and noisier**.
  
  This behavior:
  - does **not** change EM/Relaxed (both still wrong),  
  - but **reduces token-level precision** on short questions, which matches the observed drop in F1/ROUGE-L on the short-answer subset.

- In general, when the gold answer is 1–2 short sentences, any extra hallucinated bullets or repeated phrases **inflate the denominator of F1** and hurt overlap metrics.

#### 2. Sometimes more focused and closer to the gold

- **Case 4 (song “More”)**  
  - Baseline mixes 2003 and 2004 and gives a longer narrative.  
  - LoRA focuses directly on the key fact: “nine consecutive weeks in 2004” and stays more concise.
  
  Here, LoRA is actually **more aligned** with the gold description. This is consistent with the **small overall F1 gain** on the full test set: some examples do get slightly better.

#### 3. When the model is already correct, LoRA mostly preserves behavior

- **Case 5 (Hanlon’s razor)**  
  Both baseline and LoRA give the **exact same correct sentence**, so fine-tuning does not damage this knowledge.

This mix leads to what we see in the numbers:

- Some examples (like Case 4) **improve slightly**.
- Some (like Case 3) become **more verbose and less precise**.
- Some remain **unchanged** (Case 5).

On such a small test set, these effects nearly cancel, giving **very similar aggregate metrics**.

---

### Why Can Training Make Things Worse / Inconsistent?

1. **Very small evaluation set (15 examples)**  
   - One correct answer changes EM by ~0.07.  
   - With such few samples, small random differences in phrasing can move F1/ROUGE-L up or down without indicating a real, stable improvement.

2. **LoRA updates are small compared to the base model**  
   - The base model already has strong, generic instruction-following behavior.  
   - With modest LoRA rank and limited training, updates are relatively small; they shift style (more lists, more explanations) but may not significantly improve factual precision.
   - The tiny F1 change (0.3063 → 0.3215) suggests that **training did not strongly reshape** the model.

3. **Mismatch between training objective and evaluation style**  
   - If the training targets are long, explanatory answers, the model learns to produce longer, more detailed outputs.  
   - Our evaluation, especially on short factoid questions, rewards concise, tightly aligned answers.  
   - As seen in Case 3, longer outputs with extra items can hurt F1 and ROUGE-L, even if they still contain the right phrase somewhere.
---

## How to Run

This section summarizes how to set up the environment, run the baseline, train the LoRA model, generate predictions, and evaluate them.

### 1. Environment setup

Create and activate a Python environment (Python ≥ 3.10), then install dependencies:

```bash
pip install torch transformers pandas tqdm rouge-score
```
### 2. Terminal commands in operation
View in the `tests.md` file

