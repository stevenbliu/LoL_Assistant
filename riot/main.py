import os
import time
import pandas as pd
from RiotAPI import (
    get_summoner_data,
    get_ranked_match_ids,
    get_match_info,
    get_match_timeline,
)
from starter import extract_jungler_data
from RiotRateLimiter import RiotRateLimiter  # your rate limiter

OUTPUT_DIR = "database/riot_data"
MATCHES_PER_BATCH = 40
SUMMONERS_CSV = (
    "database/riot_data/found_summoners.csv"  # your summoner list with progress
)

limiter = RiotRateLimiter()

os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    # Add 'processed' column if missing
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

            dir_path = os.path.join(OUTPUT_DIR, "match_data")
            os.makedirs(dir_path, exist_ok=True)
            processed_files = set(os.listdir(dir_path))

            while True:
                match_ids = safe_request_wrapper(
                    get_ranked_match_ids,
                    puuid,
                    count=MATCHES_PER_BATCH,
                    queue_id=420,  # 420 = Ranked Solo
                )
                if not match_ids:
                    print("No more matches found for this summoner, moving on.")
                    break

                new_match_found = False

                for match_id in match_ids:
                    filename = f"{match_id}.csv"
                    output_path = os.path.join(dir_path, filename)

                    if filename in processed_files:
                        print(f"✅ Already processed match: {match_id}")
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
                        print(f"💾 Saved match data: {filename}")

                        new_match_found = True

                    except Exception as e:
                        print(f"❌ Failed to process match {match_id}: {e}")

                if not new_match_found:
                    print(
                        "No new matches processed in this batch, moving to next summoner."
                    )
                    break

                print("⏱ Waiting before next batch for this summoner...\n")
                time.sleep(30)  # adjust for rate limit and expected new matches

            # Mark summoner as processed and save progress
            df_summoners.at[idx, "processed"] = True
            df_summoners.to_csv(SUMMONERS_CSV, index=False)
            print(f"✅ Marked summoner {summoner_id} as processed.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user! Saving progress and combined data...")

    # Save combined dataset on exit
    full_path = os.path.join(OUTPUT_DIR, "all_jungler_data.csv")

    if all_jungler_dfs:
        # Load existing data if the file exists
        if os.path.exists(full_path):
            existing_df = pd.read_csv(full_path)
            print(f"📂 Loaded {len(existing_df)} existing rows from previous CSV.")
        else:
            existing_df = pd.DataFrame()

        # Concatenate new and old data
        new_data = pd.concat(all_jungler_dfs, ignore_index=True)
        full_df = pd.concat([existing_df, new_data], ignore_index=True)

        # Drop duplicates by MatchId (or MatchId + player ID if needed)
        # full_df.drop_duplicates(subset=["MatchId", ], inplace=True)
        full_df.drop_duplicates(inplace=True)

        # Save back to CSV
        full_df.to_csv(full_path, index=False)
        print(
            f"✅ Updated {full_path} with {len(new_data)} new rows, total matches: {len(full_df['MatchId'].unique())}"
        )


if __name__ == "__main__":
    main()
