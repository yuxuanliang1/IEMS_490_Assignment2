# src/peek_samples.py
from src.prepare_dataset import load_csv_with_standard_view
from transformers import AutoTokenizer
from src.data_utils import build_messages

CKPT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATA = "data/train.csv"
N = 3

ds = load_csv_with_standard_view(DATA)
tok = AutoTokenizer.from_pretrained(CKPT)

for i in range(min(N, len(ds))):
    ex = ds[i]
    msgs = build_messages(ex["instruction"], ex.get("context") or None)
    # Preview the full formatted prompt and append the starting tag for the assistant's response
    templ = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": ""}],
        tokenize=False,
        add_generation_prompt=False
    )
    print("*" * 88)
    print(f"[{i}] PROMPT >>>\n{templ[:800]}")
    print("--- GOLD RESPONSE >>>\n", ex["response"][:800])
