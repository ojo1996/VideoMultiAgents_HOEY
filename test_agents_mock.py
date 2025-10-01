#!/usr/bin/env python3
"""
Test script for agents with mock LLM responses
This allows testing the agent structure without requiring OpenAI API keys
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def create_mock_llm_agent(agent_path: str, output_path: str):
    """Create a mock version of an agent that doesn't require OpenAI API key."""
    
    # Read the original agent file
    with open(agent_path, 'r') as f:
        content = f.read()
    
    # Create mock LLM function
    mock_llm_code = '''
def call_llm(system_prompt: str, user_prompt: str, cfg=None) -> str:
    """Mock LLM that returns simple responses for testing."""
    if "decompose" in system_prompt.lower():
        return "1. Who wrote The Hobbit?\\n2. What other works did they create?"
    elif "reason" in system_prompt.lower():
        return "J.R.R. Tolkien wrote The Hobbit. He also wrote The Lord of the Rings trilogy."
    elif "finalize" in system_prompt.lower():
        return "J.R.R. Tolkien wrote The Hobbit and is also famous for The Lord of the Rings trilogy."
    elif "math" in system_prompt.lower():
        return "I need to solve this step by step. Let me write a Python script to calculate this."
    elif "video" in system_prompt.lower():
        return "Based on the video context, I can see the main action involves a person in a kitchen."
    elif "tau" in system_prompt.lower():
        return "According to the policy, I should follow the standard escalation procedure."
    else:
        return "This is a mock response for testing purposes."

def _openai_client():
    """Mock OpenAI client."""
    return None
'''
    
    # Replace the call_llm function and _openai_client
    lines = content.split('\n')
    new_lines = []
    skip_until_def = False
    
    for line in lines:
        if 'def call_llm(' in line and 'agent_systems' not in line:
            skip_until_def = True
            new_lines.append(mock_llm_code)
            continue
        elif 'def _openai_client(' in line and 'agent_systems' not in line:
            skip_until_def = True
            continue
        elif skip_until_def and line.startswith('def ') and not line.startswith('    '):
            skip_until_def = False
            new_lines.append(line)
        elif not skip_until_def:
            new_lines.append(line)
    
    # Write the mock version
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_lines))

def test_agent(agent_name: str, command: list, description: str):
    """Test a single agent with mock LLM."""
    print(f"\n{'='*60}")
    print(f"🔄 Testing {description}")
    print(f"Command: {' '.join(command)}")
    print('='*60)
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout[-300:])  # Last 300 chars
            return True
        else:
            print("❌ FAILED")
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def main():
    print("🧪 Testing Agents with Mock LLM")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("data/raw/mhqa", exist_ok=True)
    os.makedirs("data/raw/math", exist_ok=True)
    os.makedirs("data/raw/swe", exist_ok=True)
    os.makedirs("data/raw/video", exist_ok=True)
    os.makedirs("data/raw/tau", exist_ok=True)
    
    # Test MHQA LLM Agent
    print("\n🎯 Testing MHQA LLM Agent")
    success = test_agent(
        "mhqa_llm",
        ["python", "-m", "agent_systems.MHQA_agent.Main", 
         "--question", "Who wrote The Hobbit?", 
         "--out", "data/raw/mhqa/test_llm.traj.json"],
        "MHQA LLM Agent"
    )
    
    # Test MHQA Tool Agent
    print("\n🔧 Testing MHQA Tool Agent")
    success = test_agent(
        "mhqa_tool",
        ["python", "-m", "agent_systems.MHQA.main",
         "--question", "What company owns Instagram?",
         "--traj_out", "data/raw/mhqa/test_tool.traj.json",
         "--vectors_out", "data/generated_results/test_vectors.json"],
        "MHQA Tool Agent"
    )
    
    # Test Math Agent
    print("\n🧮 Testing Math Agent")
    success = test_agent(
        "math",
        ["python", "-m", "agent_systems.Math_agent.Main",
         "--question", "Solve 2x + 5 = 13",
         "--out", "data/raw/math/test.traj.json"],
        "Math Agent"
    )
    
    # Test SWE Agent
    print("\n💻 Testing SWE Agent")
    success = test_agent(
        "swe",
        ["python", "-m", "agent_systems.SWE_agent.Main",
         "--task", "Write a factorial function",
         "--out", "data/raw/swe/test.traj.json"],
        "SWE Agent"
    )
    
    # Test Video Agent
    print("\n🎥 Testing Video Agent")
    success = test_agent(
        "video",
        ["python", "-m", "agent_systems.Video_multiagent.Main",
         "--video_ctx", "data/video_samples/toy_captions.txt",
         "--question", "What happens in the video?",
         "--out", "data/raw/video/test.traj.json"],
        "Video Agent"
    )
    
    # Test TAU Agent
    print("\n📋 Testing TAU Agent")
    success = test_agent(
        "tau",
        ["python", "-m", "agent_systems.TAU_agent.Main",
         "--question", "How to handle customer complaints?",
         "--subdomain", "retail",
         "--context", "data/tau_samples/retail_policy_excerpt.md",
         "--out", "data/raw/tau/test.traj.json"],
        "TAU Agent"
    )
    
    print("\n📊 Summary")
    print("=" * 60)
    print("Check the generated files in data/raw/ to see which agents worked!")
    
    # List generated files
    for domain in ["mhqa", "math", "swe", "video", "tau"]:
        domain_dir = f"data/raw/{domain}"
        if os.path.exists(domain_dir):
            files = list(Path(domain_dir).glob("*.traj.json"))
            print(f"{domain.upper()}: {len(files)} files generated")
            for file in files:
                print(f"  ✅ {file}")

if __name__ == "__main__":
    main()
