import time
import requests
import os
import pandas as pd
from collections import deque
from RiotAPI import get_summoner_data, get_recent_match_ids, get_match_info
from RiotAPI import API_KEY

REGION = "na1"
QUEUE_TYPE = "RANKED_SOLO_5x5"
OUTPUT_CSV = "found_summoners.csv"

TIERS = ["DIAMOND", "PLATINUM"]
DIVISIONS = ["I", "II", "III", "IV"]

# Throttling setup
MAX_REQUESTS_PER_SECOND = 20
MAX_REQUESTS_PER_2_MINUTES = 100
REQUEST_WINDOW = deque()


def rate_limit():
    now = time.time()
    REQUEST_WINDOW.append(now)

    # Remove timestamps older than 2 minutes
    while REQUEST_WINDOW and now - REQUEST_WINDOW[0] > 120:
        REQUEST_WINDOW.popleft()

    # If we hit 100 requests in the last 2 minutes, wait
    if len(REQUEST_WINDOW) >= MAX_REQUESTS_PER_2_MINUTES:
        wait_time = 120 - (now - REQUEST_WINDOW[0])
        print(f"⏳ Hit 100 reqs/2min limit. Sleeping for {wait_time:.2f}s...")
        time.sleep(wait_time)

    # Always sleep a little to avoid going over 20 req/sec
    time.sleep(1 / MAX_REQUESTS_PER_SECOND)


def riot_request(url, params={}):
    headers = {"X-Riot-Token": API_KEY}
    for _ in range(3):
        rate_limit()
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            print("⏳ Rate limited (429). Waiting 2s...")
            time.sleep(2)
        elif response.ok:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            time.sleep(1)
    return None


def get_players_by_rank(tier, division, page=1):
    url = f"https://{REGION}.api.riotgames.com/lol/league/v4/entries/{QUEUE_TYPE}/{tier}/{division}"
    params = {"page": page}
    return riot_request(url, params)


def get_winrate(entry):
    wins = entry["wins"]
    losses = entry["losses"]
    return wins, wins + losses


def main():
    good_players = []

    try:
        for tier in TIERS:
            for div in DIVISIONS:
                print(f"📦 Searching {tier} {div}...")
                page = 1
                while True:
                    entries = get_players_by_rank(tier, div, page)
                    if not entries:
                        break

                    for entry in entries:
                        try:
                            summonerId = entry["summonerId"]
                            puuid = entry["puuid"]
                            wins, total = get_winrate(entry)

                            if total >= 15:
                                winrate = wins / total
                                if winrate >= 0.55:
                                    print(
                                        f"✅ SummonerId: {summonerId}: W/L: {wins}/{total} WR%:({winrate:.0%})"
                                    )
                                    good_players.append(
                                        {
                                            "summonerId": summonerId,
                                            "puuid": puuid,
                                            "tier": tier,
                                            "division": div,
                                            "winrate": round(winrate, 2),
                                            "games": total,
                                        }
                                    )
                        except Exception as e:
                            print(
                                f"❌ Failed for {entry.get('summonerName', 'UNKNOWN')}: {e}"
                            )
                            continue

                    page += 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")

    df = pd.DataFrame(good_players)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📝 Saved {len(df)} summoners to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
