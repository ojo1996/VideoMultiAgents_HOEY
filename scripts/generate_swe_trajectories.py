#!/usr/bin/env python3
"""
Generate SWE Agent Trajectories for SFT Training
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def generate_swe_trajectories(tasks: List[str], output_dir: str):
    """Generate SWE agent trajectories for given tasks."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    trajectories = []
    
    for i, task in enumerate(tasks, 1):
        print(f"Generating trajectory {i}/{len(tasks)}: {task}")
        
        # Run SWE agent
        output_file = os.path.join(output_dir, f"swe_sample_{i}.traj.json")
        
        try:
            # Run the SWE agent
            result = subprocess.run([
                "python", "-m", "agent_systems.SWE_agent.Main",
                "--task", task,
                "--out", output_file
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ Generated: {output_file}")
                trajectories.append(output_file)
            else:
                print(f"❌ Failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout: {task}")
        except Exception as e:
            print(f"💥 Error: {e}")
    
    print(f"\nGenerated {len(trajectories)} trajectories")
    return trajectories

def main():
    parser = argparse.ArgumentParser(description="Generate SWE Agent Trajectories")
    parser.add_argument("--output_dir", default="data/raw/swe", help="Output directory for trajectories")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trajectories to generate")
    args = parser.parse_args()
    
    # Sample SWE tasks
    swe_tasks = [
        "Create a Python function that calculates fibonacci numbers",
        "Write a script to find the largest number in a list",
        "Create a function that reverses a string",
        "Write a program to check if a number is prime",
        "Create a function that sorts a list of integers",
        "Write a script to count words in a text file",
        "Create a function that finds the factorial of a number",
        "Write a program to check if two strings are anagrams",
        "Create a function that removes duplicates from a list",
        "Write a script to calculate the area of a circle",
        "Create a function that checks if a string is a palindrome",
        "Write a program to find the GCD of two numbers",
        "Create a function that converts temperature from Celsius to Fahrenheit",
        "Write a script to generate random passwords",
        "Create a function that validates email addresses"
    ]
    
    # Select tasks based on num_samples
    selected_tasks = swe_tasks[:args.num_samples]
    
    print(f"Generating {len(selected_tasks)} SWE trajectories...")
    trajectories = generate_swe_trajectories(selected_tasks, args.output_dir)
    
    # Save summary
    summary = {
        "agent": "swe",
        "num_trajectories": len(trajectories),
        "tasks": selected_tasks,
        "trajectory_files": trajectories
    }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
