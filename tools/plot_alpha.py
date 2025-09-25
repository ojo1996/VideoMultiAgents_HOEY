import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob


def load_ci_data():
    """Load confidence intervals from metrics.json files"""
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
    return ci_data


def main():
    csv_path = "results/aggregate/metrics.csv"
    if not os.path.exists(csv_path):
        print("[err] results/aggregate/metrics.csv not found. Run tools/aggregate.py first.")
        return
    df = pd.read_csv(csv_path)
    ci_data = load_ci_data()
    
    os.makedirs("results/plots", exist_ok=True)
    for task, g in df.groupby("task"):
        gg = g.sort_values("alpha_task")
        plt.figure(figsize=(8, 6))
        for ts, h in gg.groupby("timestamp"):
            # Check for CI data
            has_ci = False
            for _, row in h.iterrows():
                key = f"{row['model']}_{row['alpha_task']}_{row['task']}"
                if key in ci_data:
                    has_ci = True
                    break
            
            if has_ci:
                # Plot with error bars
                alphas = h["alpha_task"].values
                scores = h["score"].values
                errors = []
                for _, row in h.iterrows():
                    key = f"{row['model']}_{row['alpha_task']}_{row['task']}"
                    if key in ci_data:
                        ci = ci_data[key]
                        error = (ci[1] - ci[0]) / 2  # half-width
                        errors.append(error)
                    else:
                        errors.append(0)
                plt.errorbar(alphas, scores, yerr=errors, marker="o", label=f"{ts} (95% CI)", capsize=5)
            else:
                plt.plot(h["alpha_task"], h["score"], marker="o", label=ts)
        
        plt.title(f"{task} vs alpha_task")
        plt.xlabel("alpha_task")
        plt.ylabel("score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = f"results/plots/{task}_alpha_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
    print("[ok] saved plots to results/plots/")


if __name__ == "__main__":
    main()


