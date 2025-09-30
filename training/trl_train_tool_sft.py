# training/trl_train_tool_sft.py
import os
import json
import argparse
import random
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import yaml


def format_example(row: Dict[str, Any]) -> str:
    """
    Turn a single SFT row into a prompt-completion string.

    Expected keys (from our SFT builder):
      - instruction: str
      - output: str
      - (optional) meta: dict
      - (optional) tool: str
    """
    instr = (row.get("instruction") or "").strip()
    out = (row.get("output") or "").strip()
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}\n"


def tokenize_function(examples, tokenizer, max_len: int):
    """
    Batch tokenizer for our JSONL dataset with columns:
      instruction, output, (optional) meta, tool
    """
    instructions = examples.get("instruction") or []
    outputs = examples.get("output") or []

    # Fallbacks if templates change names later:
    if not instructions and "prompt" in examples:
        instructions = examples["prompt"]
    if not outputs and "response" in examples:
        outputs = examples["response"]

    texts = [
        format_example({"instruction": instr, "output": out})
        for instr, out in zip(instructions, outputs)
    ]

    toks = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks


def load_jsonl(path: str):
    # Each JSON line becomes a row; keys become dataset columns.
    return load_dataset("json", data_files=path, split="train")


def main():
    ap = argparse.ArgumentParser(description="Per-tool SFT with HF Trainer")
    ap.add_argument("--config", default="configs/trl_defaults.yaml")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--tool", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow CLI overrides
    for k in ["model_name", "tool", "data_root", "out_dir"]:
        v = getattr(args, k)
        if v:
            cfg[k] = v

    model_name = cfg["model_name"]
    tool = cfg["tool"]
    data_root = cfg["data_root"]
    out_root = cfg["out_dir"]

    train_path = os.path.join(data_root, tool, cfg.get("train_file", "train.jsonl"))
    val_path = os.path.join(data_root, tool, cfg.get("val_file", "val.jsonl"))
    out_dir = os.path.join(out_root, tool)
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # ------- Load datasets -------
    ds_train = load_jsonl(train_path)
    ds_val = load_jsonl(val_path)

    # Debug: print schema and sizes so we know what we’re mapping
    print("train columns:", ds_train.column_names)
    print("val columns:", ds_val.column_names)
    print("train size:", len(ds_train), "val size:", len(ds_val))

    # ------- Tokenize -------
    max_len = int(cfg.get("max_seq_length", 1024))
    ds_train = ds_train.map(
        lambda b: tokenize_function(b, tok, max_len),
        batched=True,
        remove_columns=ds_train.column_names,
    )
    ds_val = ds_val.map(
        lambda b: tokenize_function(b, tok, max_len),
        batched=True,
        remove_columns=ds_val.column_names,
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # ------- Trainer args (robust to different transformers versions) -------
    import inspect

    ta_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=-1 if cfg.get("max_steps") in [None, "null"] else int(cfg["max_steps"]),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        eval_steps=int(cfg.get("eval_steps", 200)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=2,
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        push_to_hub=bool(cfg.get("push_to_hub", False)),
        report_to=["none"],
        do_eval=True,  # safe across versions
    )

    # Try to include evaluation_strategy if supported
    eval_strategy_cfg = cfg.get("eval_strategy", "steps")
    try:
        from transformers.trainer_utils import IntervalStrategy
        if isinstance(eval_strategy_cfg, str):
            eval_strategy_val = IntervalStrategy(eval_strategy_cfg)
        else:
            eval_strategy_val = eval_strategy_cfg
        ta_kwargs["evaluation_strategy"] = eval_strategy_val
    except Exception:
        # Older transformers: skip evaluation_strategy entirely
        pass

    # Filter by TrainingArguments signature to avoid unexpected kwargs
    TA_sig = inspect.signature(TrainingArguments)
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in TA_sig.parameters}

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate()

    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("[ok] finished SFT; saved to", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
