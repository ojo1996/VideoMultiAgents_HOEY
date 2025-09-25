import json, re, glob, os


def main():
    alpha = {}
    for d in glob.glob("merges/alpha=*/"):
        m = re.search(r"alpha=([0-9.]+)", d)
        if m:
            key = d.rstrip("/").split("/")[-1]
            alpha[key] = {"alpha_task": float(m.group(1))}
    os.makedirs(".", exist_ok=True)
    with open("alpha_settings.json", "w", encoding="utf-8") as f:
        json.dump(alpha, f, ensure_ascii=False, indent=2)
    print("[ok] wrote alpha_settings.json with", len(alpha), "entries")


if __name__ == "__main__":
    main()


