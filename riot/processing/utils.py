import os
import pandas as pd
from riot.versioning import save_metadata


def save_combined_dataset(all_dfs, output_dir):
    combined_path = os.path.join(output_dir, "all_jungler_data.csv")

    if not all_dfs:
        print("⚠️ No new data to save.")
        return

    if os.path.exists(combined_path):
        existing = pd.read_csv(combined_path)
        print(f"📂 Loaded {len(existing)} existing rows.")
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing] + all_dfs, ignore_index=True).drop_duplicates()
    combined.to_csv(combined_path, index=False)
    print(f"✅ Saved: {combined_path} | Total matches: {combined['MatchId'].nunique()}")
    save_metadata(output_dir, combined)
