import glob, json
import pandas as pd

# Read every Parquet file the run saved (large runs are split into several)
files = glob.glob("artifacts/greetings_v1/parquet*/*.parquet")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# 1. Look at the data
print(f"Rows: {len(df)} | Columns: {list(df.columns)}\n")
pd.set_option("display.max_colwidth", 45)
print(df[["language", "formality", "situation", "greeting", "english_translation"]].to_string())

# 2. Balance: did the samplers spread evenly?
print("\n--- Balance ---")
print(df["language"].value_counts().to_string())
print()
print(df["formality"].value_counts().to_string())

# 3. Duplicates
dupes = df["greeting"].duplicated().sum()
print(f"\n--- Duplicates: {dupes} of {len(df)} greetings are repeats ---")

# 4. Export to chat JSONL for fine-tuning
with open("greetings_v1.jsonl", "w") as f:
    for _, r in df.iterrows():
        f.write(json.dumps({"messages": [
            {"role": "user",
             "content": f"Write a {r['formality']} greeting in {r['language']} for {r['situation']}."},
            {"role": "assistant", "content": r["greeting"]},
        ]}, ensure_ascii=False) + "\n")
print("\nSaved greetings_v1.jsonl")
