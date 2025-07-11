import os
import time
import pandas as pd
from datetime import datetime

from RiotAPI import (
    get_summoner_data,
    get_ranked_match_ids,
    get_match_info,
    get_match_timeline,
)
from starter import extract_jungler_data
from RiotRateLimiter import RiotRateLimiter
from versioning import save_metadata

# === Configuration ===
DATA_VERSION = "v1"  # <-- update as needed
BASE_OUTPUT_DIR = os.path.join("database", "riot_data", DATA_VERSION)
MATCH_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "match_data")
SUMMONERS_CSV = os.path.join("database", "riot_data", "found_summoners.csv")
MATCHES_PER_BATCH = 40

# === Setup ===
limiter = RiotRateLimiter()
os.makedirs(MATCH_DATA_DIR, exist_ok=True)


def safe_request_wrapper(func, *args, **kwargs):
    for _ in range(5):
        try:
            limiter.wait()
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}. Retrying in 3 seconds...")
            time.sleep(3)
    raise Exception(f"Failed after retries: {func.__name__}")


def main():
    df_summoners = pd.read_csv(SUMMONERS_CSV)

    if "processed" not in df_summoners.columns:
        df_summoners["processed"] = False

    all_jungler_dfs = []

    try:
        for idx, row in df_summoners[df_summoners["processed"] == False].iterrows():
            puuid = row["puuid"]
            summoner_id = row["summonerId"]
            print(
                f"\n➡️ Processing summoner {summoner_id} ({idx+1}/{len(df_summoners)})"
            )

            processed_files = set(os.listdir(MATCH_DATA_DIR))

            while True:
                match_ids = safe_request_wrapper(
                    get_ranked_match_ids,
                    puuid,
                    count=MATCHES_PER_BATCH,
                    queue_id=420,
                )

                if not match_ids:
                    print("No more matches found for this summoner.")
                    break

                new_match_found = False

                for match_id in match_ids:
                    filename = f"{match_id}.csv"
                    output_path = os.path.join(MATCH_DATA_DIR, filename)

                    if filename in processed_files:
                        print(f"✅ Already processed: {match_id}")
                        continue

                    try:
                        match_info = safe_request_wrapper(get_match_info, match_id)
                        timeline = safe_request_wrapper(get_match_timeline, match_id)
                        participants = match_info["info"]["participants"]

                        jungler_df = extract_jungler_data(timeline, participants)
                        jungler_df["MatchId"] = match_id
                        jungler_df.to_csv(output_path, index=False)

                        all_jungler_dfs.append(jungler_df)
                        processed_files.add(filename)

                        print(f"💾 Saved: {filename}")
                        new_match_found = True
                    except Exception as e:
                        print(f"❌ Failed to process {match_id}: {e}")

                if not new_match_found:
                    print("No new matches processed in this batch.")
                    break

                print("⏱ Waiting before next batch...")
                time.sleep(30)

            df_summoners.at[idx, "processed"] = True
            df_summoners.to_csv(SUMMONERS_CSV, index=False)
            print(f"✅ Marked {summoner_id} as processed.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving...")

    # Save final combined dataset
    combined_path = os.path.join(BASE_OUTPUT_DIR, "all_jungler_data.csv")

    if all_jungler_dfs:
        if os.path.exists(combined_path):
            existing_df = pd.read_csv(combined_path)
            print(f"📂 Loaded {len(existing_df)} existing rows.")
        else:
            existing_df = pd.DataFrame()

        new_data = pd.concat(all_jungler_dfs, ignore_index=True)
        full_df = pd.concat(
            [existing_df, new_data], ignore_index=True
        ).drop_duplicates()

        full_df.to_csv(combined_path, index=False)
        print(
            f"✅ Final dataset saved: {combined_path} | Total matches: {full_df['MatchId'].nunique()}"
        )
        save_metadata(BASE_OUTPUT_DIR, full_df)


if __name__ == "__main__":
    main()
