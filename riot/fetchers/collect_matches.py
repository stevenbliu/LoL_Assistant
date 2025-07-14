import os, time
import pandas as pd
from riot.api.api_requests import (
    get_ranked_match_ids,
    get_match_info,
    get_match_timeline,
)
from riot.processing.extractor import extract_jungler_data
from riot.api.ratelimit import RiotRateLimiter
from riot.config.config import MATCH_DATA_DIR, SUMMONERS_CSV, MATCHES_PER_BATCH
import traceback

limiter = RiotRateLimiter()


def safe_request(func, *args, **kwargs):
    exponential_backoff = 1
    for _ in range(5):
        try:
            limiter.wait()
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(3 * exponential_backoff)
            exponential_backoff *= 2
    raise Exception(f"Failed after retries: {func.__name__}")


def collect_data_from_summoners():
    df = pd.read_csv(SUMMONERS_CSV)
    if "processed" not in df.columns:
        df["processed"] = False

    all_dfs = []

    try:
        for idx, row in df[df["processed"] == False].iterrows():
            puuid, sid = row["puuid"], row["leagueId"]
            print(f"\n➡️ Id: {sid} Summoner({idx+1}/{len(df)})")

            os.makedirs(MATCH_DATA_DIR, exist_ok=True)
            processed_files = set(os.listdir(MATCH_DATA_DIR))

            while True:
                match_ids = safe_request(
                    get_ranked_match_ids, puuid, MATCHES_PER_BATCH, 420
                )
                if not match_ids:
                    break

                new_found = False
                for match_id in match_ids:
                    fname = f"{match_id}.csv"
                    path = os.path.join(MATCH_DATA_DIR, fname)

                    if fname in processed_files:
                        print(f"✅ Already processed: {match_id}")
                        continue

                    try:
                        info = safe_request(get_match_info, match_id)
                        timeline = safe_request(get_match_timeline, match_id)
                        participants = info["info"]["participants"]

                        df_jungle = extract_jungler_data(timeline, participants)
                        df_jungle["MatchId"] = match_id
                        df_jungle.to_csv(path, index=False)

                        all_dfs.append(df_jungle)
                        processed_files.add(fname)
                        new_found = True
                        print(f"💾 Saved: {fname}")
                    except Exception as e:
                        print(f"❌ Failed: {e}")
                        traceback.print_exc()

                if not new_found:
                    break
                time.sleep(30)

            df.at[idx, "processed"] = True
            df.to_csv(SUMMONERS_CSV, index=False)
            print(f"✅ Marked {sid} as processed.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")
        df.to_csv(SUMMONERS_CSV, index=False)

    return all_dfs
