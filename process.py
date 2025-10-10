import os
import pandas as pd
import numpy as np

# === INPUT ===
FREE_TEXT_FILE = "free-text.csv"
FIXED_TEXT_FILE = "fixed-text.csv"
OUTPUT_DIR = "users"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === READ FILES ===
df_free = pd.read_csv(FREE_TEXT_FILE, low_memory=False)
df_fixed = pd.read_csv(FIXED_TEXT_FILE, low_memory=False)

# clean column names (remove trailing spaces)
df_free.columns = df_free.columns.str.strip()
df_fixed.columns = df_fixed.columns.str.strip()

print("Free text columns:", df_free.columns.tolist())
print("Fixed text columns:", df_fixed.columns.tolist())

print("process free file")

# --- FREE TEXT ---
free_features = [
    "DU.key1.key1",
    "DD.key1.key2",
    "DU.key1.key2",
    "UD.key1.key2",
    "UU.key1.key2"
]

for user_id, group in df_free.groupby("participant"):
    with open(os.path.join(OUTPUT_DIR, f"{user_id}.txt"), "w") as user_file:
        for _, row in group.iterrows():
            feats = pd.to_numeric(row[free_features], errors="coerce").values
            if not np.isnan(feats).any() and len(feats) == 5:
                user_file.write(" ".join(map(str, feats)) + "\n")

print("process fixed file")

# --- FIXED TEXT ---
fixed_features = [c for c in df_fixed.columns if c not in ["participant", "session", "repetition", "total time"]]

for user_id, group in df_fixed.groupby("participant"):
    with open(os.path.join(OUTPUT_DIR, f"{user_id}.txt"), "a") as user_file:  # append
        for _, row in group.iterrows():
            feats = pd.to_numeric(row[fixed_features], errors="coerce").values
            # chunk into groups of 5
            for i in range(0, len(feats), 5):
                group5 = feats[i:i+5]
                if len(group5) == 5 and not np.isnan(group5).any():
                    user_file.write(" ".join(map(str, group5)) + "\n")

print(f"✅ Done! User files are in: {OUTPUT_DIR}/")
