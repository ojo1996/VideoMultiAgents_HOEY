# scripts/preview_sft.py
import json, glob, os, argparse

def read_jsonl(path, limit=3):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sft")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    for tool_dir in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(tool_dir): continue
        name = os.path.basename(tool_dir)
        trn = os.path.join(tool_dir, "train.jsonl")
        if not os.path.exists(trn): continue
        print(f"\n=== {name} ===")
        for row in read_jsonl(trn, limit=args.limit):
            print(json.dumps(row, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()