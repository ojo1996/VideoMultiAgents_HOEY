#!/usr/bin/env python3
"""
Build SWE Tool-Specific SFT Datasets
Groups trajectories by tool and creates SFT training data
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

def load_trajectory(path: str) -> Dict[str, Any]:
    """Load a trajectory file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_tool_calls(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from trajectory steps."""
    tool_calls = []
    
    for step in trajectory.get("steps", []):
        if step.get("role") == "assistant":
            content = step.get("content", "")
            phase = step.get("phase", "")
            
            # Extract bash commands
            if "```bash" in content:
                bash_start = content.find("```bash") + 7
                bash_end = content.find("```", bash_start)
                if bash_end > bash_start:
                    bash_cmd = content[bash_start:bash_end].strip()
                    tool_calls.append({
                        "tool": "bash",
                        "phase": phase,
                        "command": bash_cmd,
                        "context": content,
                        "turn_id": step.get("turn_id", 0)
                    })
            
            # Extract file operations
            if "WRITE:" in content or "file" in content.lower():
                # Look for file creation/editing patterns
                if "cat >" in content or "write" in content.lower():
                    tool_calls.append({
                        "tool": "file_edit",
                        "phase": phase,
                        "command": "file_edit",
                        "context": content,
                        "turn_id": step.get("turn_id", 0)
                    })
    
    return tool_calls

def create_sft_example(tool_call: Dict[str, Any], trajectory: Dict[str, Any]) -> Dict[str, str]:
    """Create SFT training example from tool call."""
    
    tool = tool_call["tool"]
    phase = tool_call["phase"]
    context = tool_call["context"]
    command = tool_call["command"]
    
    # Create instruction based on tool and phase
    if tool == "bash":
        instruction = f"Execute the following bash command for {phase.lower()} phase: {command}"
        response = f"```bash\n{command}\n```"
    elif tool == "file_edit":
        instruction = f"Perform file editing operation for {phase.lower()} phase"
        response = context
    
    # Format as conversation
    text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    
    return {
        "text": text,
        "instruction": instruction,
        "response": response,
        "tool": tool,
        "phase": phase,
        "domain": trajectory.get("domain", "swe"),
        "success": trajectory.get("success", False),
        "task_id": trajectory.get("task_id", "unknown")
    }

def build_tool_datasets(trajectory_paths: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Build datasets grouped by tool."""
    
    tool_datasets = defaultdict(list)
    
    for traj_path in trajectory_paths:
        try:
            trajectory = load_trajectory(traj_path)
            tool_calls = extract_tool_calls(trajectory)
            
            for tool_call in tool_calls:
                sft_example = create_sft_example(tool_call, trajectory)
                tool_datasets[tool_call["tool"]].append(sft_example)
                
        except Exception as e:
            print(f"Error processing {traj_path}: {e}")
            continue
    
    return dict(tool_datasets)

def save_tool_datasets(tool_datasets: Dict[str, List[Dict[str, str]]], output_dir: str):
    """Save tool-specific datasets."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for tool, examples in tool_datasets.items():
        if not examples:
            continue
            
        # Save as JSONL for training
        output_file = os.path.join(output_dir, f"swe_{tool}_sft.jsonl")
        with open(output_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        
        # Save summary
        summary = {
            "tool": tool,
            "num_examples": len(examples),
            "phases": list(set(ex["phase"] for ex in examples)),
            "success_rate": sum(1 for ex in examples if ex["success"]) / len(examples)
        }
        
        summary_file = os.path.join(output_dir, f"swe_{tool}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ {tool}: {len(examples)} examples -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Build SWE Tool-Specific SFT Datasets")
    parser.add_argument("--trajectory_glob", default="data/raw/swe/*.traj.json", help="Glob pattern for trajectory files")
    parser.add_argument("--output_dir", default="data/sft/swe", help="Output directory for SFT datasets")
    args = parser.parse_args()
    
    # Find trajectory files
    import glob
    trajectory_paths = glob.glob(args.trajectory_glob)
    print(f"Found {len(trajectory_paths)} trajectory files")
    
    if not trajectory_paths:
        print("No trajectory files found!")
        return
    
    # Build tool datasets
    print("Building tool-specific datasets...")
    tool_datasets = build_tool_datasets(trajectory_paths)
    
    # Save datasets
    print("Saving datasets...")
    save_tool_datasets(tool_datasets, args.output_dir)
    
    # Overall summary
    total_examples = sum(len(examples) for examples in tool_datasets.values())
    print(f"\n📊 Summary:")
    print(f"Total examples: {total_examples}")
    for tool, examples in tool_datasets.items():
        print(f"  {tool}: {len(examples)} examples")

if __name__ == "__main__":
    main()
