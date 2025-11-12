# src/prepare_dataset.py
from typing import Optional, Dict, Any
import ast
import pandas as pd
from datasets import Dataset

def _get_from_seq(cell: Any) -> Optional[str]:
    # Safely parse here to extract the first value
    if not isinstance(cell, str) or not cell.strip():
        return None
    try:
        obj = ast.literal_eval(cell)
    except Exception:
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if isinstance(obj, dict):
        val = obj.get("value")
        if isinstance(val, list) and val:
            return str(val[0])
        if isinstance(val, str):
            return val
    return None

def add_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    o_instr = pick("original-instruction")
    o_ctx   = pick("original-context")
    o_resp  = pick("original-response")
    n_instr = pick("new-instruction")
    n_ctx   = pick("new-context")
    n_resp  = pick("new-response")

    inst_list, ctx_list, resp_list = [], [], []
    for _, row in df.iterrows():
        instr = row[o_instr] if o_instr else None
        ctx   = row[o_ctx]   if o_ctx   else None
        resp  = row[o_resp]  if o_resp  else None

        if (not isinstance(instr, str) or not instr.strip()) and n_instr:
            instr = _get_from_seq(row[n_instr])
        if (not isinstance(ctx, str) or not ctx.strip()) and n_ctx:
            ctx   = _get_from_seq(row[n_ctx])
        if (not isinstance(resp, str) or not resp.strip()) and n_resp:
            resp  = _get_from_seq(row[n_resp])

        inst_list.append("" if instr is None else str(instr))
        ctx_list.append(""   if ctx   is None else str(ctx))
        resp_list.append(""  if resp  is None else str(resp))

    df = df.copy()
    df["instruction"] = inst_list
    df["context"]     = ctx_list
    df["response"]    = resp_list
    return df

def load_csv_with_standard_view(path: str) -> Dataset:
    pdf = pd.read_csv(path)
    pdf = add_standard_columns(pdf)
    pdf = pdf[(pdf["instruction"].str.strip() != "") & (pdf["response"].str.strip() != "")]
    return Dataset.from_pandas(pdf, preserve_index=False)
