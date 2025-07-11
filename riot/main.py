import os
import time
import pandas as pd
from RiotAPI import (
    get_summoner_data,
    get_recent_match_ids,
    get_match_info,
    get_match_timeline,
)
from starter import extract_jungler_data
from RiotRateLimiter import RiotRateLimiter  # 👈 Import your rate limiter

OUTPUT_DIR = "database/riot_data"
MATCHES_PER_BATCH = 10

limiter = RiotRateLimiter()  # 👈 Global limiter

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_request_wrapper(func, *args, **kwargs):
    for _ in range(5):  # Retry up to 5 times
        try:
            limiter.wait()
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}. Retrying in 3 seconds...")
            time.sleep(3)
    raise Exception(f"Failed after retries: {func.__name__}")


def main():
    summoner_name = "Zdev#1111"
    game_name, tag_line = summoner_name.split("#")
    print(f"Fetching data for summoner: {summoner_name}")

    summoner_data = safe_request_wrapper(get_summoner_data, game_name, tag_line)
    puuid = summoner_data["puuid"]

    dir_path = os.path.join(OUTPUT_DIR, "match_data")
    os.makedirs(dir_path, exist_ok=True)

    processed_ids = set(os.listdir(dir_path))

    all_jungler_dfs = []

    try:
        while True:
            match_ids = safe_request_wrapper(
                get_recent_match_ids, puuid, count=MATCHES_PER_BATCH
            )

            print(f"\n🧠 Retrieved {len(match_ids)} recent match IDs...")

            for match_id in match_ids:
                filename = f"{match_id}.csv"
                output_path = os.path.join(dir_path, filename)

                if filename in processed_ids:
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
                    processed_ids.add(filename)
                    print(f"💾 Saved: {filename}")

                except Exception as e:
                    print(f"❌ Failed to process {match_id}: {e}")

            print("⏱ Waiting before next batch...\n")
            time.sleep(
                30
            )  # Adjust this depending on how frequently new games are expected

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving final combined dataset...")

    # Final save
    print("all_jungler_dfs length:", len(all_jungler_dfs))
    if all_jungler_dfs:
        full_df = pd.concat(all_jungler_dfs, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, "all_jungler_data.csv")
        full_df.to_csv(output_path, index=False)
        print(f"✅ Combined data saved to: {output_path}")


if __name__ == "__main__":
    main()
