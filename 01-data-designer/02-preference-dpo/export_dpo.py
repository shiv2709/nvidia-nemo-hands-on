import glob, json
import pandas as pd

files = glob.glob("artifacts/prefs_v1/parquet*/*.parquet")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"Total pairs: {len(df)}")

print("\nJudge verdicts:")
print(df["winner"].value_counts().to_string())

# Keep only clear winners (drop ties)
clear = df[df["winner"].isin(["A", "B"])]
print(f"\nKept {len(clear)} clear pairs, dropped {len(df) - len(clear)} ties")

with open("prefs_v1_dpo.jsonl", "w") as f:
    for _, r in clear.iterrows():
        f.write(json.dumps({
            "prompt": r["question"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }, ensure_ascii=False) + "\n")
print("Saved prefs_v1_dpo.jsonl")
