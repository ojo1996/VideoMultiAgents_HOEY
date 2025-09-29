# agent_systems/MHQA_agent/Tools.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

# ---------- OpenAI client ----------

def _openai_client():
    """
    Requires:
      - pip install --upgrade openai
      - env var OPENAI_API_KEY set
    """
    from openai import OpenAI
    return OpenAI()

@dataclass
class LLMConfig:
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

def call_llm(system_prompt: str, user_prompt: str, cfg: LLMConfig) -> str:
    client = _openai_client()
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""

# ---------- Trajectory helpers ----------

def new_trajectory(task_id: str, domain: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "domain": domain,
        "question": None,
        "actions": [],
        "final_answer": None,
        "metadata": {"start_ts": time.time()},
    }

def record_action(traj: Dict[str, Any], tool: str, inp: Any, out: Any):
    step = len(traj["actions"]) + 1
    traj["actions"].append({"step": step, "tool": tool, "input": inp, "output": out})

def save_trajectory(traj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    traj["metadata"]["end_ts"] = time.time()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)

# ---------- MHQA prompts ----------

DECOMPOSE_SYSTEM = (
    "You are a careful multi-hop question decomposition assistant.\n"
    "Given a question, produce 2–5 short sub-questions that, if answered in\n"
    "sequence, would yield the final answer. Avoid redundancy and keep each hop atomic.\n"
    "Return as a numbered list."
)

HOP_REASON_SYSTEM = (
    "You are a concise reasoning assistant. Given a sub-question and any prior findings,\n"
    "write a brief evidence-based note (2–4 sentences) and end with a one-line provisional answer.\n"
    "Do not include unrelated background. Be decisive."
)

FINALIZE_SYSTEM = (
    "You are an answer synthesizer. Using the original question and the sequence of hop notes,\n"
    "write a single, direct final answer. If uncertain, state the best-supported answer\n"
    "and briefly note residual uncertainty in parentheses."
)