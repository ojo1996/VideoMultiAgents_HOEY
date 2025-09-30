# tools/validate_traj.py
import argparse, json, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified_glob", default="data/unified/*/*.json")
    args = ap.parse_args()
    files = glob.glob(args.unified_glob)
    bad = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                _ = json.load(f)
        except Exception as e:
            bad += 1
            print("[bad]", fp, "->", e)
    print(f"[ok] scanned {len(files)} files; bad={bad}")

if __name__ == "__main__":
    main()