import json, glob
import sys


def read(p):
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return None


def main(tol: float = 0.05):
    base = None
    one = None
    for j in glob.glob("results/batch/*/alpha=*/metrics.json"):
        d = read(j)
        if not d:
            continue
        a = (d.get("alpha") or {}).get("alpha_task")
        if a == 0.0:
            base = d["metrics"]
        if a == 1.0:
            one = d["metrics"]
    if base is None or one is None:
        print("[warn] missing alpha=0.0 or alpha=1.0 in results; skipping check")
        return 0
    bad = []
    for k, v in one.items():
        b = base.get(k)
        if b is None:
            continue
        # pass if difference within tolerance
        if abs(v - b) > tol:
            bad.append((k, b, v))
    if bad:
        print("[fail] anchor check exceeded tolerance:")
        for k, b, v in bad:
            print(f"  {k}: alpha=0.0 {b} vs alpha=1.0 {v}")
        return 2
    print("[ok] anchor check passed within tolerance")
    return 0


if __name__ == "__main__":
    tol = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    raise SystemExit(main(tol))


