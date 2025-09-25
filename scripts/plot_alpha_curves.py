import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Plot or print metric vs alpha curves")
    ap.add_argument("--results_root", required=True, help="Folder containing per-alpha subfolders with results.json")
    ap.add_argument("--dataset", default=None, help="Task name inside results.json, if following lm-eval format")
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if not name.startswith("alpha"):
            continue
        try:
            alpha = float(name.replace("alpha", ""))
        except ValueError:
            continue
        # Try several result file conventions
        candidates = [
            sub / "results.json",
            sub / f"{name}__{args.dataset}__results.json" if args.dataset else sub / "_no_file_",
        ]
        metric = None
        for p in candidates:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # lm-eval style
                if "results" in data and args.dataset and args.dataset in data["results"]:
                    # pick the first metric available
                    md = data["results"][args.dataset]
                    for k, v in md.items():
                        if k.endswith(",none"):
                            metric = v
                            break
                # direct metric
                if metric is None and isinstance(data, dict) and "metric" in data:
                    metric = data["metric"]
        if metric is not None:
            rows.append((alpha, metric))

    if not rows:
        print("[warn] no results found")
        return

    rows.sort(key=lambda x: x[0])
    try:
        import matplotlib.pyplot as plt
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("alpha")
        plt.ylabel("metric")
        plt.title("Score vs Alpha")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = root / "alpha_curve.png"
        plt.savefig(out_png, dpi=150)
        print(f"[ok] wrote {out_png}")
    except Exception:
        print("alpha\tmetric")
        for a, m in rows:
            print(f"{a}\t{m}")


if __name__ == "__main__":
    main()


