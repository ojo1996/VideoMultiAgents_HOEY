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

# ---------- Reward calculation ----------

def calculate_trajectory_reward(trajectory: Dict[str, Any]) -> Dict[str, float]:
    """Calculate reward for MHQA trajectory."""
    rewards = {}
    
    # 1. Task completion reward
    success = trajectory.get('success', False)
    final_answer = trajectory.get('final_answer', '')
    rewards['completion'] = 1.0 if success else (0.5 if final_answer and len(final_answer.strip()) > 5 else 0.0)
    
    # 2. Tool usage efficiency
    actions = trajectory.get('actions', [])
    tool_actions = [a for a in actions if a.get('tool') not in ['reason', 'finalize', 'decompose', 'reason_hop']]
    tool_count = len(tool_actions)
    rewards['efficiency'] = max(0.0, 1.0 - (tool_count - 1) * 0.1) if tool_count > 0 else 0.5
    
    # 3. Answer quality (MHQA-specific)
    rewards['quality'] = evaluate_mhqa_answer_quality(final_answer)
    
    # 4. Reasoning quality
    reasoning_actions = [a for a in actions if a.get('tool') in ['reason', 'reason_hop']]
    rewards['reasoning'] = evaluate_reasoning_quality(reasoning_actions)
    
    # 5. Multi-hop structure
    rewards['multi_hop'] = evaluate_multi_hop_structure(actions)
    
    # Weighted total
    weights = {'completion': 0.4, 'efficiency': 0.2, 'quality': 0.3, 'reasoning': 0.1, 'multi_hop': 0.0}
    rewards['total'] = sum(rewards[key] * weights[key] for key in weights)
    
    return rewards

def evaluate_mhqa_answer_quality(answer: str) -> float:
    """Evaluate MHQA answer quality."""
    if not answer or len(answer.strip()) < 5:
        return 0.0
    
    score = 0.5
    
    # Check for multi-hop reasoning indicators
    if any(phrase in answer.lower() for phrase in ['first', 'then', 'next', 'finally', 'based on']):
        score += 0.2
    
    # Check for detailed explanation
    if answer.count('.') > 2:
        score += 0.2
    
    # Check for evidence or citations
    if any(indicator in answer.lower() for indicator in ['according to', 'evidence', 'shows that']):
        score += 0.1
    
    return min(1.0, score)

def evaluate_reasoning_quality(reasoning_actions: List[Dict[str, Any]]) -> float:
    """Evaluate reasoning step quality."""
    if not reasoning_actions:
        return 0.0
    
    total_score = 0.0
    for action in reasoning_actions:
        output = action.get('output', '')
        if len(output) < 10:
            continue
        
        score = 0.0
        
        # Check for reasoning indicators
        reasoning_indicators = ['because', 'therefore', 'since', 'thus', 'hence', 'so']
        if any(indicator in output.lower() for indicator in reasoning_indicators):
            score += 0.3
        
        # Check for step-by-step structure
        if any(char.isdigit() for char in output) and ('.' in output or '\n' in output):
            score += 0.2
        
        # Check for logical connectors
        connectors = ['and', 'or', 'but', 'however', 'moreover', 'furthermore']
        if any(connector in output.lower() for connector in connectors):
            score += 0.1
        
        total_score += min(1.0, score)
    
    return total_score / len(reasoning_actions)

def evaluate_multi_hop_structure(actions: List[Dict[str, Any]]) -> float:
    """Evaluate if trajectory follows proper multi-hop structure."""
    decompose_actions = [a for a in actions if a.get('tool') == 'decompose']
    reason_hop_actions = [a for a in actions if a.get('tool') == 'reason_hop']
    finalize_actions = [a for a in actions if a.get('tool') == 'finalize']
    
    # Should have decompose, multiple reason_hop, and finalize
    if not decompose_actions or not finalize_actions:
        return 0.0
    
    if len(reason_hop_actions) < 2:
        return 0.5  # Partial credit for some hops
    
    return 1.0  # Full credit for proper multi-hop structure

def save_trajectory_with_reward(traj: Dict[str, Any], path: str):
    """Save trajectory with calculated reward."""
    reward = calculate_trajectory_reward(traj)
    traj['reward'] = reward
    save_trajectory(traj, path)

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
