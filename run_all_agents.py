#!/usr/bin/env python3
"""
Comprehensive Multi-Agent Trajectory Generation
Runs all agent types end-to-end with real datasets and questions
"""

import os
import json
import subprocess
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

def run_command(cmd: List[str], description: str, timeout: int = 300) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
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

def create_directories():
    """Create necessary directories for all agents."""
    dirs = [
        "data/raw/mhqa", "data/raw/math", "data/raw/swe", "data/raw/video", "data/raw/tau",
        "data/unified/mhqa", "data/unified/math", "data/unified/swe", "data/unified/video", "data/unified/tau",
        "data/sft", "data/rewards", "models/sft", "models/rl", "runs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ All directories created")

def generate_mhqa_trajectories(num_trajectories: int = 10):
    """Generate MHQA trajectories using both LLM and tool-based agents."""
    print(f"\n🎯 Generating {num_trajectories} MHQA Trajectories")
    
    # Questions from config
    questions = [
        "Who wrote The Hobbit and what other famous works did they create?",
        "What is the capital of France and what is its population?",
        "Who was the first person to walk on the moon and when did it happen?",
        "What is the largest planet in our solar system and what are its main characteristics?",
        "Who painted the Mona Lisa and in which museum is it located?",
        "What company owns Instagram and who founded that company?",
        "Who directed the movie Titanic and what other famous movies did they direct?",
        "What is the chemical symbol for gold and what are its main properties?",
        "Who invented the telephone and what other inventions are they known for?",
        "What is the tallest mountain in the world and in which country is it located?"
    ]
    
    trajectory_files = []
    
    # Generate with LLM-based MHQA agent
    for i, question in enumerate(questions[:num_trajectories//2]):
        output_file = f"data/raw/mhqa/llm_sample_{i+1}.traj.json"
        
        success = run_command([
            "python", "-m", "agent_systems.MHQA_agent.Main",
            "--question", question,
            "--out", output_file
        ], f"MHQA LLM Agent - Question {i+1}")
        
        if success and os.path.exists(output_file):
            trajectory_files.append(output_file)
            print(f"✅ Generated LLM trajectory: {output_file}")
    
    # Generate with tool-based MHQA agent
    for i, question in enumerate(questions[num_trajectories//2:num_trajectories]):
        output_file = f"data/raw/mhqa/tool_sample_{i+1}.traj.json"
        vectors_file = f"data/generated_results/mhqa_vectors_{i+1}.json"
        
        success = run_command([
            "python", "-m", "agent_systems.MHQA.main",
            "--question", question,
            "--traj_out", output_file,
            "--vectors_out", vectors_file
        ], f"MHQA Tool Agent - Question {i+1}")
        
        if success and os.path.exists(output_file):
            trajectory_files.append(output_file)
            print(f"✅ Generated Tool trajectory: {output_file}")
    
    return trajectory_files

def generate_math_trajectories(num_trajectories: int = 10):
    """Generate Math agent trajectories."""
    print(f"\n🧮 Generating {num_trajectories} Math Trajectories")
    
    questions = [
        "Solve the equation 2x + 5 = 13",
        "What is the derivative of x^2 + 3x + 1?",
        "Calculate the area of a circle with radius 5",
        "What is the value of sin(π/2)?",
        "Find the integral of 2x from 0 to 3",
        "Solve the quadratic equation x^2 - 5x + 6 = 0",
        "What is the limit of (x^2 - 1)/(x - 1) as x approaches 1?",
        "Calculate the volume of a sphere with radius 3",
        "What is the derivative of ln(x)?",
        "Find the sum of the first 10 natural numbers"
    ]
    
    trajectory_files = []
    
    for i, question in enumerate(questions[:num_trajectories]):
        output_file = f"data/raw/math/sample_{i+1}.traj.json"
        
        success = run_command([
            "python", "-m", "agent_systems.Math_agent.Main",
            "--question", question,
            "--out", output_file
        ], f"Math Agent - Question {i+1}")
        
        if success and os.path.exists(output_file):
            trajectory_files.append(output_file)
            print(f"✅ Generated Math trajectory: {output_file}")
    
    return trajectory_files

def generate_swe_trajectories(num_trajectories: int = 10):
    """Generate SWE agent trajectories."""
    print(f"\n💻 Generating {num_trajectories} SWE Trajectories")
    
    questions = [
        "Write a Python function to calculate the factorial of a number",
        "Create a simple web server using Flask",
        "Implement a binary search algorithm",
        "Write a function to reverse a string",
        "Create a class to represent a bank account",
        "Write a function to find the maximum element in a list",
        "Implement a simple calculator with basic operations",
        "Write a function to check if a string is a palindrome",
        "Create a simple file reader that counts lines",
        "Implement a basic sorting algorithm"
    ]
    
    trajectory_files = []
    
    for i, question in enumerate(questions[:num_trajectories]):
        output_file = f"data/raw/swe/sample_{i+1}.traj.json"
        
        success = run_command([
            "python", "-m", "agent_systems.SWE_agent.Main",
            "--question", question,
            "--out", output_file
        ], f"SWE Agent - Question {i+1}")
        
        if success and os.path.exists(output_file):
            trajectory_files.append(output_file)
            print(f"✅ Generated SWE trajectory: {output_file}")
    
    return trajectory_files

def generate_video_trajectories(num_trajectories: int = 5):
    """Generate Video agent trajectories."""
    print(f"\n🎥 Generating {num_trajectories} Video Trajectories")
    
    questions = [
        "What happens in the first scene of the video?",
        "Who are the main characters shown in the video?",
        "What is the main action taking place in the video?",
        "What objects can you see in the video?",
        "What is the setting or location of the video?"
    ]
    
    trajectory_files = []
    
    for i, question in enumerate(questions[:num_trajectories]):
        output_file = f"data/raw/video/sample_{i+1}.traj.json"
        
        success = run_command([
            "python", "-m", "agent_systems.Video_multiagent.Main",
            "--video_ctx", "data/video_samples/toy_captions.txt",
            "--question", question,
            "--out", output_file
        ], f"Video Agent - Question {i+1}")
        
        if success and os.path.exists(output_file):
            trajectory_files.append(output_file)
            print(f"✅ Generated Video trajectory: {output_file}")
    
    return trajectory_files

def generate_tau_trajectories(num_trajectories: int = 5):
    """Generate TAU agent trajectories."""
    print(f"\n📋 Generating {num_trajectories} TAU Trajectories")
    
    questions = [
        "How should I handle a customer complaint about a delayed order?",
        "What are the security protocols for accessing sensitive data?",
        "What is the procedure for requesting time off?",
        "How should I escalate a technical issue to the engineering team?",
        "What is the policy for handling confidential information?"
    ]
    
    trajectory_files = []
    
    for i, question in enumerate(questions[:num_trajectories]):
        output_file = f"data/raw/tau/sample_{i+1}.traj.json"
        
        success = run_command([
            "python", "-m", "agent_systems.TAU_agent.Main",
            "--question", question,
            "--subdomain", "retail",
            "--context", "data/tau_samples/retail_policy_excerpt.md",
            "--out", output_file
        ], f"TAU Agent - Question {i+1}")
        
        if success and os.path.exists(output_file):
            trajectory_files.append(output_file)
            print(f"✅ Generated TAU trajectory: {output_file}")
    
    return trajectory_files

def convert_trajectories_to_unified(domain: str, trajectory_files: List[str]):
    """Convert trajectories to unified format."""
    print(f"\n🔄 Converting {domain} trajectories to unified format")
    
    unified_files = []
    
    for traj_file in trajectory_files:
        if not os.path.exists(traj_file):
            print(f"⚠️  Skipping missing file: {traj_file}")
            continue
            
        output_file = f"data/unified/{domain}/{Path(traj_file).stem}.json"
        
        success = run_command([
            "python", "convert_to_out_traj_format.py",
            traj_file,
            "--domain", domain,
            "--out_dir", f"data/unified/{domain}"
        ], f"Converting {Path(traj_file).name}")
        
        if success and os.path.exists(output_file):
            unified_files.append(output_file)
            print(f"✅ Converted: {output_file}")
    
    return unified_files

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Multi-Agent Trajectory Generation")
    parser.add_argument("--num_trajectories", type=int, default=10, help="Number of trajectories per agent")
    parser.add_argument("--agents", nargs="+", default=["mhqa", "math", "swe", "video", "tau"], 
                       help="Which agents to run")
    parser.add_argument("--skip_generation", action="store_true", help="Skip trajectory generation")
    parser.add_argument("--skip_conversion", action="store_true", help="Skip trajectory conversion")
    args = parser.parse_args()
    
    print("🚀 Comprehensive Multi-Agent Trajectory Generation")
    print("=" * 60)
    
    # Check environment
    print("\n📋 Environment Check:")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Create directories
    create_directories()
    
    all_trajectory_files = {}
    
    if not args.skip_generation:
        # Generate trajectories for each agent
        if "mhqa" in args.agents:
            all_trajectory_files["mhqa"] = generate_mhqa_trajectories(args.num_trajectories)
        
        if "math" in args.agents:
            all_trajectory_files["math"] = generate_math_trajectories(args.num_trajectories)
        
        if "swe" in args.agents:
            all_trajectory_files["swe"] = generate_swe_trajectories(args.num_trajectories)
        
        if "video" in args.agents:
            all_trajectory_files["video"] = generate_video_trajectories(args.num_trajectories)
        
        if "tau" in args.agents:
            all_trajectory_files["tau"] = generate_tau_trajectories(args.num_trajectories)
    
    if not args.skip_conversion:
        # Convert all trajectories to unified format
        for domain, traj_files in all_trajectory_files.items():
            if traj_files:
                convert_trajectories_to_unified(domain, traj_files)
    
    # Summary
    print("\n📊 Summary")
    print("=" * 60)
    for domain, traj_files in all_trajectory_files.items():
        print(f"{domain.upper()}: {len(traj_files)} trajectories generated")
        for traj_file in traj_files:
            if os.path.exists(traj_file):
                print(f"  ✅ {traj_file}")
            else:
                print(f"  ❌ {traj_file} (missing)")
    
    print("\n🎉 Multi-agent trajectory generation complete!")
    print("Next steps:")
    print("1. Review generated trajectories in data/raw/")
    print("2. Check unified format in data/unified/")
    print("3. Run training: python training/sft_train_3b_final.py --trajectory_glob 'data/raw/*/*.traj.json'")

if __name__ == "__main__":
    main()
