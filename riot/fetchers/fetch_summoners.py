import time
import requests
import pandas as pd
from riot.api.ratelimit import RiotRateLimiter
import riot.config.config as config  # your config.py with API_KEY, TIERS, DIVISIONS, REGION, etc.

API_KEY = config.RIOT_API_KEY
REGION = config.REGION
QUEUE_TYPE = config.QUEUE_TYPE
TIERS = config.TIERS
DIVISIONS = config.DIVISIONS

limiter = RiotRateLimiter(
    per_second=config.MAX_REQUESTS_PER_SECOND,
    per_2min=config.MAX_REQUESTS_PER_2_MINUTES,
)


def riot_request(url, params=None):
    headers = {"X-Riot-Token": API_KEY}
    if params is None:
        params = {}

    for _ in range(3):
        limiter.wait()
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
    wins = entry.get("wins", 0)
    losses = entry.get("losses", 0)
    return wins, wins + losses


def find_good_summoners(min_games=15, min_winrate=0.55):
    good_players = []
    try:
        for tier in TIERS:
            for division in DIVISIONS:
                print(f"📦 Searching {tier} {division}...")
                page = 1

                while True:
                    entries = get_players_by_rank(tier, division, page)
                    if not entries:
                        print(
                            f"No more entries found for {tier} {division} page {page}."
                        )
                        break

                    for entry in entries:
                        try:
                            # print(f"Processing SummonerId: {entry}")

                            leagueId = entry["leagueId"]
                            puuid = entry["puuid"]
                            wins, total = get_winrate(entry)

                            if total >= min_games:
                                winrate = wins / total
                                if winrate >= min_winrate:
                                    print(
                                        f"✅ leagueId: {leagueId}: W/L: {wins}/{total} WR%: {winrate:.0%}"
                                    )
                                    good_players.append(
                                        {
                                            "leagueId": leagueId,
                                            "puuid": puuid,
                                            "tier": tier,
                                            "division": division,
                                            "winrate": round(winrate, 2),
                                            "games": total,
                                        }
                                    )
                        except Exception as e:
                            print(
                                f"❌ Failed for entry {entry.get('leagueId', 'UNKNOWN')}: {e}"
                            )
                            continue

                    page += 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")

    return good_players
