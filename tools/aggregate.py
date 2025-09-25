import json, glob, re, csv, os
import jsonschema
from pathlib import Path


def validate_metrics_schema(data, schema_path="schemas/metrics_schema.json"):
    """Validate metrics.json against schema"""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(data, schema)
        return True
    except Exception as e:
        print(f"[schema error] {e}")
        return False


def main():
    rows = []
    schema_path = Path("schemas/metrics_schema.json")
    if not schema_path.exists():
        print("[warn] schema not found, skipping validation")
    
    for p in glob.glob("results/batch/*/alpha=*/metrics.json"):
        try:
            run = json.load(open(p, "r", encoding="utf-8"))
            # Validate schema if available
            if schema_path.exists() and not validate_metrics_schema(run, str(schema_path)):
                print(f"[skip] invalid schema in {p}")
                continue
        except Exception as e:
            print(f"[skip] failed to load {p}: {e}")
            continue
        
        a = (run.get("alpha") or {}).get("alpha_task")
        parts = p.split("/")
        ts = parts[2] if len(parts) > 3 else ""
        model = parts[3] if len(parts) > 4 else ""
        
        # Extract all alpha values for multi-alpha support
        alpha_dict = run.get("alpha", {})
        alpha_task = alpha_dict.get("alpha_task")
        alpha_bash = alpha_dict.get("alpha_bash")
        alpha_read = alpha_dict.get("alpha_read")
        alpha_write = alpha_dict.get("alpha_write")
        
        for task, val in (run.get("metrics") or {}).items():
            # Skip hierarchical keys for CSV (keep flat ones)
            if "/" in task:
                continue
            rows.append({
                "timestamp": ts, 
                "model": model, 
                "alpha_task": alpha_task,
                "alpha_bash": alpha_bash,
                "alpha_read": alpha_read,
                "alpha_write": alpha_write,
                "task": task, 
                "score": val
            })
    
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


