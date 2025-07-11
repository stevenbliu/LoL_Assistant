import os
from RiotAPI import (
    get_summoner_data,
    get_recent_match_ids,
    get_match_info,
    get_match_timeline,
)

from starter import extract_jungler_data

import pandas as pd

OUTPUT_DIR = "database/riot_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    summoner_name = "Zdev#1111"
    game_name, tag_line = summoner_name.split("#")

    print(f"Fetching data for summoner: {summoner_name}")
    summoner_data = get_summoner_data(game_name, tag_line)
    puuid = summoner_data["puuid"]

    match_ids = get_recent_match_ids(puuid, count=20)  # Fetch 20 recent matches

    if not match_ids:
        print("No recent matches found. Exiting.")
        return

    all_jungler_dfs = []
    dir_path = os.path.join(OUTPUT_DIR, "match_data")
    os.makedirs(dir_path, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    total_matches = len(match_ids)
    print(f"Found {total_matches} matches to process.")

    for idx, match_id in enumerate(match_ids, start=1):
        output_path = os.path.join(dir_path, f"{match_id}.csv")

        if os.path.exists(output_path):
            print(
                f"[{idx}/{total_matches}] Skipping already processed match: {match_id}"
            )
            skipped_count += 1
            continue

        try:
            match_info = get_match_info(match_id)
            timeline = get_match_timeline(match_id)
            participants = match_info["info"]["participants"]
            jungler_df = extract_jungler_data(timeline, participants)

            jungler_df["MatchId"] = match_id  # Tag the match

            jungler_df.to_csv(output_path, index=False)
            all_jungler_dfs.append(jungler_df)

            processed_count += 1
            print(f"[{idx}/{total_matches}] Saved: {output_path}")
        except Exception as e:
            failed_count += 1
            print(f"[{idx}/{total_matches}] Failed to process match {match_id}: {e}")

    print(f"\nSummary:")
    print(f"Processed matches: {processed_count}")
    print(f"Skipped matches: {skipped_count}")
    print(f"Failed matches: {failed_count}")

    # Optional: merge all into one DataFrame
    if all_jungler_dfs:
        full_df = pd.concat(all_jungler_dfs, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, "all_jungler_data.csv")
        full_df.to_csv(output_path, index=False)
        print("Saved combined dataset as all_jungler_data.csv")


if __name__ == "__main__":
    main()
