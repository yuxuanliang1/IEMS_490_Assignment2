# split_data.py
import os
import csv
import json
import pandas as pd
from datasets import load_dataset

# Set parameter
DATASET = "argilla/databricks-dolly-15k-curated-en"
SPLIT = "train"
OUT_DIR = "data"
SEED = 42
N_TEST = 15
N_TRAIN = 1000
CSV_ENCODING = "utf-8-sig"

ds = load_dataset(DATASET, split=SPLIT)
n = len(ds)
print(f"Loaded {DATASET} ({SPLIT}) with {n} rows. Columns: {ds.column_names}")

# Shuffle using a fixed random seed
shuf = ds.shuffle(seed=SEED)

# Train
take_test = min(N_TEST, n)
test_ds = shuf.select(range(take_test))

# Test
remaining = n - take_test
take_train = min(N_TRAIN, remaining)
train_ds = shuf.select(range(take_test, take_test + take_train))

os.makedirs(OUT_DIR, exist_ok=True)

def save_all(dset, name):
    pdf = dset.to_pandas()

    # CSV
    pdf.to_csv(os.path.join(OUT_DIR, f"{name}.csv"),
               index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # Jsonl
    pdf.to_json(os.path.join(OUT_DIR, f"{name}.jsonl"),
                orient="records", lines=True, force_ascii=False)

save_all(train_ds, "train")
save_all(test_ds,  "test")

print("Done.")
