# agent_systems/Video_multiagent/Tools.py
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

# ---------- Context loader ----------

def load_video_context(path: str) -> str:
    """
    Accepts:
      - .txt: free-form transcript/captions/notes
      - .json: either a string or list[str] of captions/events; we join with newlines
    Returns a single normalized string.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        if path.lower().endswith(".json"):
            data = json.load(f)
            if isinstance(data, str):
                return data.strip()
            if isinstance(data, list):
                return "\n".join(str(x).strip() for x in data if str(x).strip())
            # generic JSON: make a shallow, readable dump
            return json.dumps(data, ensure_ascii=False, indent=2)
        return f.read().strip()

# ---------- Prompts ----------

VIDEO_REASON_SYSTEM = (
    "You are a precise video-understanding assistant working from textual context "
    "(transcript, frame captions, shot list). Answer questions using only the provided context. "
    "Cite brief evidence snippets in parentheses when helpful. If the answer is not supported, say so."
)

VIDEO_FINALIZE_SYSTEM = (
    "You are finalizing a video QA answer. Using the question and your preliminary notes, "
    "write a single, direct answer. If uncertain, state the best-supported answer and note uncertainty "
    "in parentheses."
)