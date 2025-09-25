import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = "results/aggregate/metrics.csv"
    if not os.path.exists(csv_path):
        print("[err] results/aggregate/metrics.csv not found. Run tools/aggregate.py first.")
        return
    df = pd.read_csv(csv_path)
    os.makedirs("results/plots", exist_ok=True)
    for task, g in df.groupby("task"):
        gg = g.sort_values("alpha_task")
        plt.figure()
        for ts, h in gg.groupby("timestamp"):
            plt.plot(h["alpha_task"], h["score"], marker="o", label=ts)
        plt.title(f"{task} vs alpha_task")
        plt.xlabel("alpha_task")
        plt.ylabel("score")
        plt.grid(True)
        plt.legend()
        out = f"results/plots/{task}_alpha_curve.png"
        plt.savefig(out, dpi=150)
        plt.close()
    print("[ok] saved plots to results/plots/")


if __name__ == "__main__":
    main()


