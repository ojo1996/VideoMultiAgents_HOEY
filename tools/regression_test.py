#!/usr/bin/env python3
"""
Quick regression test for CI: runs tiny eval with alpha=0,1 on CPU
"""
import subprocess
import sys
import tempfile
import os
from pathlib import Path


def main():
    print("[*] Running regression test (alpha=0,1 on CPU)...")
    
    # Create tiny test config
    test_config = {
        "seeds": [1],
        "device": "cpu",
        "tasks": {
            "test_task": {
                "loader": "agent_systems/Math_agent/main.py",
                "metric": "exact_match"
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml
        yaml.dump(test_config, f)
        test_config_path = f.name
    
    try:
        # Test alpha=0 (should match base)
        print("[*] Testing alpha=0...")
        rc0 = subprocess.run([
            "python", "scripts/apply_vector_and_eval.py",
            "--use_mergekit",
            "--mk_base", "Qwen/Qwen2.5-3B",
            "--mk_rl", "models/AFM-MHQA-Agent-3B-rl", 
            "--mk_sft", "models/AFM-MHQA-Agent-3B-sft",
            "--alpha_task", "0.0",
            "--out_dir", "test_alpha0",
            "--eval_task", "test_task"
        ], capture_output=True, text=True)
        
        # Test alpha=1 (should match donor)
        print("[*] Testing alpha=1...")
        rc1 = subprocess.run([
            "python", "scripts/apply_vector_and_eval.py",
            "--use_mergekit",
            "--mk_base", "Qwen/Qwen2.5-3B",
            "--mk_rl", "models/AFM-MHQA-Agent-3B-rl",
            "--mk_sft", "models/AFM-MHQA-Agent-3B-sft", 
            "--alpha_task", "1.0",
            "--out_dir", "test_alpha1",
            "--eval_task", "test_task"
        ], capture_output=True, text=True)
        
        if rc0.returncode != 0:
            print(f"[fail] alpha=0 test failed: {rc0.stderr}")
            return 1
        if rc1.returncode != 0:
            print(f"[fail] alpha=1 test failed: {rc1.stderr}")
            return 1
            
        print("[ok] regression test passed")
        return 0
        
    finally:
        # Cleanup
        os.unlink(test_config_path)
        import shutil
        for d in ["test_alpha0", "test_alpha1"]:
            if os.path.exists(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    sys.exit(main())
