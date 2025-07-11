import os
from RiotAPI import (
    get_summoner_data,
    get_recent_match_ids,
    get_match_info,
    get_match_timeline,
)

from starter import extract_jungler_data

# from preprocessing import preprocess_for_training

import pandas as pd

OUTPUT_DIR = "riot_data/match_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    summoner_name = "Zdev#1111"
    game_name, tag_line = summoner_name.split("#")

    print(f"Fetching data for summoner: {summoner_name}")
    summoner_data = get_summoner_data(game_name, tag_line)
    puuid = summoner_data["puuid"]

    match_ids = get_recent_match_ids(puuid, count=20)  # Fetch 20 recent matches

    all_jungler_dfs = []

    for match_id in match_ids:
        output_path = os.path.join(OUTPUT_DIR, f"{match_id}.csv")
        if os.path.exists(output_path):
            print(f"Skipping already processed match: {match_id}")
            continue

        try:
            match_info = get_match_info(match_id)
            timeline = get_match_timeline(match_id)
            participants = match_info["info"]["participants"]
            jungler_df = extract_jungler_data(timeline, participants)

            jungler_df["MatchId"] = match_id  # Tag the match

            jungler_df.to_csv(output_path, index=False)
            all_jungler_dfs.append(jungler_df)

            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to process match {match_id}: {e}")

    # Optional: merge all into one DataFrame
    if all_jungler_dfs:
        full_df = pd.concat(all_jungler_dfs, ignore_index=True)
        full_df.to_csv("all_jungler_data.csv", index=False)
        print("Saved combined dataset as all_jungler_data.csv")


if __name__ == "__main__":
    main()
