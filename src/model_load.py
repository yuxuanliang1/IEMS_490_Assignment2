# src/model_loader.py
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_CKPT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

def load_chat_model(
    checkpoint: str = DEFAULT_CKPT,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if device_map:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=dtype, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=dtype
        ).to(device)
    model.eval()
    return tok, model
