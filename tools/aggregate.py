import json, glob, re, csv, os


def main():
    rows = []
    for p in glob.glob("results/batch/*/alpha=*/metrics.json"):
        try:
            run = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        a = (run.get("alpha") or {}).get("alpha_task")
        parts = p.split("/")
        ts = parts[2] if len(parts) > 3 else ""
        model = parts[3] if len(parts) > 4 else ""
        for task, val in (run.get("metrics") or {}).items():
            rows.append({"timestamp": ts, "model": model, "alpha_task": a, "task": task, "score": val})
    os.makedirs("results/aggregate", exist_ok=True)
    out = "results/aggregate/metrics.csv"
    if not rows:
        print("[warn] no metrics found")
        return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ok] wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()


