#!/usr/bin/env python3
"""
Reward Calculation Utilities for Trajectories

This script provides comprehensive reward calculation functions for different
domains and can be used to analyze trajectory quality and generate reward datasets.
"""

import json
import argparse
import glob
from typing import Dict, Any, List, Tuple
from pathlib import Path
import re
import numpy as np


class RewardCalculator:
    """Comprehensive reward calculator for agent trajectories."""
    
    def __init__(self):
        self.domain_weights = {
            'mhqa': {'completion': 0.4, 'efficiency': 0.2, 'quality': 0.3, 'reasoning': 0.1},
            'video': {'completion': 0.3, 'efficiency': 0.2, 'quality': 0.4, 'reasoning': 0.1},
            'math': {'completion': 0.5, 'efficiency': 0.1, 'quality': 0.3, 'reasoning': 0.1},
            'swe': {'completion': 0.4, 'efficiency': 0.3, 'quality': 0.2, 'reasoning': 0.1},
            'tau': {'completion': 0.3, 'efficiency': 0.2, 'quality': 0.4, 'reasoning': 0.1},
        }
    
    def calculate_reward(self, trajectory: Dict[str, Any], domain: str = None) -> Dict[str, float]:
        """
        Calculate comprehensive reward for a trajectory.
        
        Args:
            trajectory: The trajectory dictionary
            domain: The domain (auto-detected if not provided)
        
        Returns:
            Dictionary with individual reward components and total
        """
        if domain is None:
            domain = trajectory.get('domain', 'unknown')
        
        rewards = {}
        
        # 1. Task completion reward
        rewards['completion'] = self._calculate_completion_reward(trajectory)
        
        # 2. Tool usage efficiency
        rewards['efficiency'] = self._calculate_efficiency_reward(trajectory)
        
        # 3. Answer quality (domain-specific)
        rewards['quality'] = self._calculate_quality_reward(trajectory, domain)
        
        # 4. Reasoning quality
        rewards['reasoning'] = self._calculate_reasoning_reward(trajectory)
        
        # 5. Conciseness reward
        rewards['conciseness'] = self._calculate_conciseness_reward(trajectory)
        
        # 6. Tool selection appropriateness
        rewards['tool_selection'] = self._calculate_tool_selection_reward(trajectory, domain)
        
        # Calculate weighted total
        weights = self.domain_weights.get(domain, self.domain_weights['mhqa'])
        rewards['total'] = sum(rewards[key] * weights.get(key, 0.0) for key in weights)
        
        return rewards
    
    def _calculate_completion_reward(self, trajectory: Dict[str, Any]) -> float:
        """Calculate reward based on task completion."""
        success = trajectory.get('success', False)
        final_answer = trajectory.get('final_answer', '')
        
        if success:
            return 1.0
        
        # Partial credit for having a final answer
        if final_answer and len(final_answer.strip()) > 5:
            return 0.5
        
        return 0.0
    
    def _calculate_efficiency_reward(self, trajectory: Dict[str, Any]) -> float:
        """Calculate reward based on tool usage efficiency."""
        actions = trajectory.get('actions', [])
        tool_actions = [a for a in actions if a.get('tool') not in ['reason', 'finalize']]
        tool_count = len(tool_actions)
        
        # Optimal range: 1-3 tools
        if tool_count == 0:
            return 0.0
        elif tool_count <= 3:
            return 1.0
        else:
            # Penalty for excessive tool usage
            return max(0.0, 1.0 - (tool_count - 3) * 0.2)
    
    def _calculate_quality_reward(self, trajectory: Dict[str, Any], domain: str) -> float:
        """Calculate domain-specific quality reward."""
        final_answer = trajectory.get('final_answer', '')
        
        if not final_answer or len(final_answer.strip()) < 5:
            return 0.0
        
        base_score = 0.5
        
        if domain == 'math':
            return self._evaluate_math_quality(final_answer)
        elif domain == 'mhqa':
            return self._evaluate_mhqa_quality(final_answer)
        elif domain == 'video':
            return self._evaluate_video_quality(final_answer)
        elif domain == 'swe':
            return self._evaluate_swe_quality(final_answer)
        elif domain == 'tau':
            return self._evaluate_tau_quality(final_answer)
        else:
            return base_score
    
    def _evaluate_math_quality(self, answer: str) -> float:
        """Evaluate mathematical answer quality."""
        score = 0.5
        
        # Check for mathematical reasoning
        math_indicators = ['equation', 'solve', 'calculate', 'formula', 'theorem', 'proof']
        if any(indicator in answer.lower() for indicator in math_indicators):
            score += 0.2
        
        # Check for numerical content
        if re.search(r'\d+', answer):
            score += 0.2
        
        # Check for step-by-step structure
        if re.search(r'\d+\.', answer) or 'step' in answer.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_mhqa_quality(self, answer: str) -> float:
        """Evaluate multi-hop QA answer quality."""
        score = 0.5
        
        # Check for multi-hop reasoning indicators
        if any(phrase in answer.lower() for phrase in ['first', 'then', 'next', 'finally']):
            score += 0.2
        
        # Check for detailed explanation
        if answer.count('.') > 2:
            score += 0.2
        
        # Check for evidence or citations
        if any(indicator in answer.lower() for indicator in ['according to', 'based on', 'evidence']):
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_video_quality(self, answer: str) -> float:
        """Evaluate video QA answer quality."""
        score = 0.5
        
        # Check for video-specific content
        video_indicators = ['video', 'scene', 'frame', 'action', 'character', 'visual']
        if any(indicator in answer.lower() for indicator in video_indicators):
            score += 0.3
        
        # Check for temporal references
        if any(indicator in answer.lower() for indicator in ['before', 'after', 'during', 'while']):
            score += 0.2
        
        return min(1.0, score)
    
    def _evaluate_swe_quality(self, answer: str) -> float:
        """Evaluate software engineering answer quality."""
        score = 0.5
        
        # Check for code content
        if '```' in answer or any(keyword in answer for keyword in ['def ', 'class ', 'function']):
            score += 0.3
        
        # Check for error handling
        if 'error' not in answer.lower() and 'bug' not in answer.lower():
            score += 0.1
        
        # Check for best practices
        if any(practice in answer.lower() for practice in ['test', 'documentation', 'clean code']):
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_tau_quality(self, answer: str) -> float:
        """Evaluate TAU (policy/tool-aware) answer quality."""
        score = 0.5
        
        # Check for policy awareness
        policy_indicators = ['policy', 'rule', 'guideline', 'procedure']
        if any(indicator in answer.lower() for indicator in policy_indicators):
            score += 0.2
        
        # Check for tool awareness
        tool_indicators = ['tool', 'method', 'approach', 'technique']
        if any(indicator in answer.lower() for indicator in tool_indicators):
            score += 0.2
        
        # Check for structured response
        if any(struct in answer.lower() for struct in ['step', 'process', 'workflow']):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_reasoning_reward(self, trajectory: Dict[str, Any]) -> float:
        """Calculate reward based on reasoning quality."""
        actions = trajectory.get('actions', [])
        reasoning_actions = [a for a in actions if a.get('tool') == 'reason']
        
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
            if re.search(r'\d+\.', output) or 'step' in output.lower():
                score += 0.2
            
            # Check for logical connectors
            connectors = ['and', 'or', 'but', 'however', 'moreover', 'furthermore']
            if any(connector in output.lower() for connector in connectors):
                score += 0.1
            
            # Check for question analysis
            if any(word in output.lower() for word in ['what', 'how', 'why', 'when', 'where']):
                score += 0.1
            
            total_score += min(1.0, score)
        
        return total_score / len(reasoning_actions)
    
    def _calculate_conciseness_reward(self, trajectory: Dict[str, Any]) -> float:
        """Calculate reward based on response conciseness."""
        actions = trajectory.get('actions', [])
        total_length = sum(len(str(a.get('output', ''))) for a in actions)
        
        # Optimal length range: 100-2000 characters
        if total_length < 100:
            return 0.5  # Too short
        elif total_length <= 2000:
            return 1.0  # Optimal
        else:
            # Penalty for excessive length
            return max(0.0, 1.0 - (total_length - 2000) / 10000)
    
    def _calculate_tool_selection_reward(self, trajectory: Dict[str, Any], domain: str) -> float:
        """Calculate reward based on appropriate tool selection."""
        actions = trajectory.get('actions', [])
        tools_used = [a.get('tool') for a in actions if a.get('tool') not in ['reason', 'finalize']]
        
        if not tools_used:
            return 0.5  # Neutral if no tools used
        
        # Domain-specific tool appropriateness
        domain_tools = {
            'mhqa': ['search', 'retrieve', 'read'],
            'video': ['load_context', 'analyze'],
            'math': ['calculate', 'solve'],
            'swe': ['bash', 'write', 'read'],
            'tau': ['load_context', 'analyze']
        }
        
        appropriate_tools = domain_tools.get(domain, [])
        if not appropriate_tools:
            return 0.5
        
        # Check if used tools are appropriate for domain
        appropriate_count = sum(1 for tool in tools_used if any(apt in tool.lower() for apt in appropriate_tools))
        return appropriate_count / len(tools_used)


def analyze_trajectory_rewards(trajectory_path: str, domain: str = None) -> Dict[str, Any]:
    """Analyze rewards for a single trajectory file."""
    with open(trajectory_path, 'r', encoding='utf-8') as f:
        trajectory = json.load(f)
    
    calculator = RewardCalculator()
    rewards = calculator.calculate_reward(trajectory, domain)
    
    return {
        'file': trajectory_path,
        'task_id': trajectory.get('task_id', 'unknown'),
        'domain': trajectory.get('domain', 'unknown'),
        'success': trajectory.get('success', False),
        'rewards': rewards
    }


def batch_analyze_rewards(trajectory_glob: str, output_path: str = None) -> List[Dict[str, Any]]:
    """Analyze rewards for multiple trajectory files."""
    trajectory_paths = glob.glob(trajectory_glob)
    results = []
    
    print(f"Analyzing {len(trajectory_paths)} trajectory files...")
    
    for path in trajectory_paths:
        try:
            result = analyze_trajectory_rewards(path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate rewards for trajectories")
    parser.add_argument("--trajectory_glob", default="data/raw/*.traj.json", 
                       help="Glob pattern for trajectory files")
    parser.add_argument("--output", default="data/rewards/analysis.json",
                       help="Output file for reward analysis")
    parser.add_argument("--domain", default=None,
                       help="Override domain for all trajectories")
    parser.add_argument("--min_reward", type=float, default=0.0,
                       help="Minimum total reward threshold")
    parser.add_argument("--stats", action="store_true",
                       help="Print reward statistics")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Analyze rewards
    results = batch_analyze_rewards(args.trajectory_glob, args.output)
    
    if args.stats:
        # Print statistics
        total_rewards = [r['rewards']['total'] for r in results]
        completion_rewards = [r['rewards']['completion'] for r in results]
        quality_rewards = [r['rewards']['quality'] for r in results]
        
        print(f"\nReward Statistics:")
        print(f"Total trajectories: {len(results)}")
        print(f"Average total reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        print(f"Average completion reward: {np.mean(completion_rewards):.3f} ± {np.std(completion_rewards):.3f}")
        print(f"Average quality reward: {np.mean(quality_rewards):.3f} ± {np.std(quality_rewards):.3f}")
        
        # Filter by minimum reward
        filtered_results = [r for r in results if r['rewards']['total'] >= args.min_reward]
        print(f"Trajectories above {args.min_reward} reward: {len(filtered_results)}")
        
        # Domain breakdown
        domains = {}
        for result in results:
            domain = result['domain']
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(result['rewards']['total'])
        
        print(f"\nDomain breakdown:")
        for domain, rewards in domains.items():
            print(f"  {domain}: {len(rewards)} trajectories, avg reward: {np.mean(rewards):.3f}")


if __name__ == "__main__":
    main()
