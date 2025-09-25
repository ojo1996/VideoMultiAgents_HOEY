# agent_systems/Math_agent/Tools.py
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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

# ---------- Shell runner ----------

def run_bash(cmd: str, timeout: int = 60) -> Tuple[int, str, str]:
    """Run a shell command cross-platform, return (rc, stdout, stderr)."""
    if platform.system().lower().startswith("win"):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    else:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

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
    traj["metadata"]["end_ts"] = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2, ensure_ascii=False)

# ---------- Prompts ----------

OS_HINT = "Windows (cmd)" if platform.system() == "Windows" else "Unix-like (bash)"

MATH_SYSTEM_PROMPT = (
    f"You are a careful Math assistant on a restricted machine. The host OS is {OS_HINT}.\n"
    "Your job: compute the final numeric answer to the user's math question.\n"
    "Output EXACTLY ONE fenced bash block containing ONE command of the form:\n\n"
    "THOUGHT: <1-2 line reasoning>\n"
    "```bash\n"
    "python -c \"print(<PURE_PY_EXPRESSION>)\"\n"
    "```\n\n"
    "Rules:\n"
    "- Use only pure Python arithmetic (integers, floats, **, // as needed). No imports, no files, no I/O.\n"
    "- The command must produce ONLY the final number on stdout.\n"
    "- Do not explain outside the bash block.\n"
)

FINALIZE_MATH_PROMPT = (
    "You are finalizing a math task. The latest command output is the final numeric result.\n"
    "Respond with ONE sentence of the form: \"Answer: N\" (replace N with the number).\n"
)