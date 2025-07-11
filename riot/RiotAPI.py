import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd  # Import pandas

# Get the root directory (parent of the script's parent folder)
root_dir = Path(__file__).resolve().parent.parent  # goes two levels up
# os.environ.clear()

# Load .env from the root directory
load_dotenv(dotenv_path=root_dir / ".env")

API_KEY = os.getenv("RIOT_DEV_API_KEY")
if not API_KEY:
    raise ValueError("Missing RIOT_DEV_API_KEY in environment variables")
print("API Key:", API_KEY)  # Debugging line to check if the key is loaded

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": API_KEY.strip(),
}
REGION = "na1"
MATCH_REGION = "americas"


def get_summoner_data(game_name, tag_line):
    url = f"https://{MATCH_REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def get_ranked_match_ids(puuid, count=1, queue_id=420):  # 420 = Ranked Solo
    url = (
        f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/"
        f"{puuid}/ids?count={count}&queue={queue_id}"
    )
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def get_match_timeline(match_id):
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def get_match_info(match_id):
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()
