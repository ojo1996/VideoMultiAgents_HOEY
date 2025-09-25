import json, re, glob, os


def main():
    alpha = {}
    for d in glob.glob("merges/*/"):
        name = os.path.basename(d.rstrip("/"))
        kvs = {}
        # Parse multi-alpha folder names: alpha=0.5_alpha_bash=1.0_alpha_read=0.0
        for k, v in re.findall(r"([a-zA-Z0-9_]+)=([0-9.]+)", name):
            if k.startswith("alpha"):
                kvs[k] = float(v)
        if kvs:
            alpha[name] = kvs
    os.makedirs(".", exist_ok=True)
    with open("alpha_settings.json", "w", encoding="utf-8") as f:
        json.dump(alpha, f, ensure_ascii=False, indent=2)
    print("[ok] wrote alpha_settings.json with", len(alpha), "entries")
    if alpha:
        print("[*] discovered alpha patterns:")
        for name, alphas in alpha.items():
            print(f"  {name}: {alphas}")


if __name__ == "__main__":
    main()


