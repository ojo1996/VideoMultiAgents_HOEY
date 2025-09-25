import pandas as pd
import json
import glob
import os


def main():
    """Generate paper-ready CSV with mean, lo, hi columns"""
    csv_path = "results/aggregate/metrics.csv"
    if not os.path.exists(csv_path):
        print("[err] results/aggregate/metrics.csv not found. Run tools/aggregate.py first.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Load CI data
    ci_data = {}
    for p in glob.glob("results/batch/*/alpha=*/metrics.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            parts = p.split("/")
            model = parts[3] if len(parts) > 4 else ""
            alpha = data.get("alpha", {}).get("alpha_task")
            if alpha is not None and "metrics_ci" in data:
                for task, ci in data["metrics_ci"].items():
                    key = f"{model}_{alpha}_{task}"
                    ci_data[key] = ci
        except Exception:
            continue
    
    # Add CI columns
    df['lo'] = None
    df['hi'] = None
    
    for idx, row in df.iterrows():
        key = f"{row['model']}_{row['alpha_task']}_{row['task']}"
        if key in ci_data:
            ci = ci_data[key]
            df.at[idx, 'lo'] = ci[0]
            df.at[idx, 'hi'] = ci[1]
    
    # Save paper CSV
    os.makedirs("results/paper", exist_ok=True)
    paper_csv = "results/paper/alpha_sweep_data.csv"
    df.to_csv(paper_csv, index=False)
    print(f"[ok] wrote paper CSV to {paper_csv}")
    
    # Print summary
    print("[*] Summary by task:")
    for task, group in df.groupby("task"):
        print(f"  {task}: {len(group)} points, alpha range [{group['alpha_task'].min():.1f}, {group['alpha_task'].max():.1f}]")


if __name__ == "__main__":
    main()
