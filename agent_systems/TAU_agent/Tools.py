# agent_systems/TAU_agent/Tools.py
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

@dataclass
class LLMConfig:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

def _openai_client():
    from openai import OpenAI
    return OpenAI()

def call_llm(system_prompt: str, user_prompt: str, cfg: Optional[LLMConfig] = None) -> str:
    cfg = cfg or LLMConfig()
    client = _openai_client()
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def run_bash(cmd: str, timeout: int = 60) -> Tuple[int, str, str]:
    if platform.system().lower().startswith("win"):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    else:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def new_trajectory(task_id: str, domain: str) -> Dict[str, Any]:
    return {"task_id": task_id, "domain": domain, "question": None,
            "actions": [], "final_answer": "", "metadata": {"start_ts": time.time()}}

def record_action(traj: Dict[str, Any], tool: str, input_obj: Any, output_obj: Any) -> None:
    step = len(traj["actions"]) + 1
    traj["actions"].append({
        "step": step, "tool": tool,
        "input": input_obj, "output": output_obj,
    })

def save_trajectory(traj: Dict[str, Any], out_path: str) -> None:
    traj["metadata"]["end_ts"] = time.time()
    write_json(out_path, traj)

TAU_SYSTEM_PROMPT = """You are a TAU-bench style assistant. 
Follow the context rules strictly, answer concisely, and cite short evidence if needed."""

FINALIZE_SYSTEM_PROMPT = """You are finalizing an answer for a standardized trajectory.
Rewrite the answer so it is clear, self-contained, and compliant with context rules."""