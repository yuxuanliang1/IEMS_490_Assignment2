## 1) Run the baseline
```bash
python -m src.baseline_greedy \
  --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --test_csv data/test.csv \
  --out_file logs/baseline_greedy_newtok.jsonl \
  --max_prompt_len 1024 \
  --max_new_tokens 256
```
## 1) Train the LoRA model
```bath
python -m src.train_lora \
  --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --train_csv data/train.csv \
  --eval_csv  data/test.csv \
  --output_dir models/lora-smol2-1k-v2 \
  --max_seq_len 768 \
  --epochs 2 \
  --bs 1 --ga 24 \
  --lr 5e-5 \
  --lora_dropout 0.1 \
  --lora_r 8
```

## 3) Generate predictions with the LoRA model
```bath
python -m src.baseline_greedy \
  --model_name models/lora-smol2-1k-v2 \
  --test_csv data/test.csv \
  --out_file logs/lora_v2_greedy_newtok.jsonl \
  --max_prompt_len 1024 \
  --max_new_tokens 256
```
## 4) Evaluate predictions
#### Baseline
```bath
python -m src.eval_accuracy_plus \
  --pred_file logs/baseline_greedy_newtok.jsonl \
  --short_thresh 15 \
  --rouge_thresh 0.5
```
#### LoRA model
```bath
python -m src.eval_accuracy_plus \
  --pred_file logs/lora_v2_greedy_newtok.jsonl \
  --short_thresh 15 \
  --rouge_thresh 0.5
```