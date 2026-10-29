import glob
import pandas as pd

df = pd.concat([pd.read_parquet(f) for f in
                glob.glob("artifacts/prefs_v1/parquet*/*.parquet")], ignore_index=True)

df["len_a"] = df["response_a"].str.split().str.len()
df["len_b"] = df["response_b"].str.split().str.len()
df["longer_won"] = ((df["winner"] == "A") & (df["len_a"] > df["len_b"])) | \
                   ((df["winner"] == "B") & (df["len_b"] > df["len_a"]))

print(f"Rows analysed: {len(df)}")
print(f"Avg words  A (Lightning): {df['len_a'].mean():.0f}   B (Super): {df['len_b'].mean():.0f}")
print(f"Longer answer won: {df['longer_won'].sum()}/{len(df)} ({df['longer_won'].mean():.0%})")
print(f"Answers over 120 words  A: {(df['len_a'] > 120).sum()}   B: {(df['len_b'] > 120).sum()}")

chosen_len = df[df["winner"].isin(["A", "B"])]["chosen"].str.split().str.len()
print(f"Chosen answers over 120 words: {(chosen_len > 120).sum()}")

print("\nB (Super) win rate by audience:")
print((df["winner"] == "B").groupby(df["audience"]).mean().round(2).to_string())
