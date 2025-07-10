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


def get_recent_match_ids(puuid, count=1):
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}"
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


def extract_jungler_data(timeline, participants):
    junglers = [p for p in participants if p["individualPosition"].upper() == "JUNGLE"]
    # Sort or order by participantId to keep consistent order
    junglers = sorted(junglers, key=lambda p: p["participantId"])

    # Collect all timestamps from frames (in seconds)
    timestamps = []
    for frame in timeline["info"]["frames"]:
        timestamp_ms = frame["timestamp"]
        seconds = timestamp_ms // 1000
        timestamps.append(seconds)
        # timestamps.append(timestamp_ms)

    # Initialize dict to hold per-jungler data keyed by timestamp (seconds)
    # data = {"Minute": [f"{t // 60}:{t % 60:02d}" for t in timestamps]}
    # data = {[t for t in timestamps]}
    data = {"Second": timestamps}

    # For each jungler, collect lists for each stat, keyed by participantId
    for jungler in junglers:
        pid = str(jungler["participantId"])
        player_name = (
            jungler.get("summonerName")
            or jungler.get("riotIdGameName")
            or f"Player{pid}"
        )
        champion_name = jungler["championName"]
        position = jungler["individualPosition"]
        team = jungler["teamId"]

        minions_killed_list = []
        jungle_minions_killed_list = []
        x_list = []
        y_list = []

        for frame in timeline["info"]["frames"]:
            p_data = frame["participantFrames"][pid]
            minions_killed_list.append(p_data.get("minionsKilled", 0))
            jungle_minions_killed_list.append(p_data.get("jungleMinionsKilled", 0))
            pos = p_data.get("position", {"x": None, "y": None})
            x_list.append(pos.get("x"))
            y_list.append(pos.get("y"))

        # Add columns per jungler with player name and stat
        # prefix = f"{player_name} ({champion_name})"
        prefix = pid
        data[f"{prefix}_Player"] = [player_name] * len(timestamps)
        data[f"{prefix}_Champion"] = [champion_name] * len(timestamps)
        data[f"{prefix}_MinionsKilled"] = minions_killed_list
        data[f"{prefix}_JungleMinionsKilled"] = jungle_minions_killed_list
        data[f"{prefix}_X"] = x_list
        data[f"{prefix}_Y"] = y_list
        data[f"{prefix}_Team"] = [team] * len(timestamps)
        data[f"{prefix}_Position"] = [position] * len(timestamps)

    df = pd.DataFrame(data)
    return df


def print_keys(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            print(path)
            if isinstance(v, dict) or (
                isinstance(v, list) and v and isinstance(v[0], dict)
            ):
                # For lists of dicts, print keys from the first item
                if isinstance(v, list):
                    print_keys(v[0], path + "[0]")
                else:
                    print_keys(v, path)


def main():
    summoner_name = "Zdev#1111"
    game_name, tag_line = summoner_name.split("#")
    # game_name, tag_line = "Zdev", "1111"

    print(f"Fetching data for summoner: {summoner_name}")
    summoner_data = get_summoner_data(game_name, tag_line)
    puuid = summoner_data["puuid"]

    match_ids = get_recent_match_ids(puuid)
    match_id = match_ids[0]
    match_info = get_match_info(match_id)
    timeline = get_match_timeline(match_id)

    unique_events = set()
    for frame in timeline["info"]["frames"]:
        if "events" in frame:
            for event in frame["events"]:
                if (
                    event["type"] == "MONSTER_KILL"
                    or event["type"] == "ELITE_MONSTER_KILL"
                ):
                    print(f"Timestamp: {event['timestamp']}ms")
                    print(f"Monster Type: {event.get('monsterType', 'N/A')}")
                    print(f"Killer Participant ID: {event.get('killerId', 'N/A')}")
                    print(f"Monster Subtype: {event.get('monsterSubType', 'N/A')}")
                    print("---")
                # if event not in unique_events:
                unique_events.add(event["type"])
    print(f"Unique events count: {unique_events}")

    # print_keys(match_info)
    # print_keys(timeline)

    participants = match_info["info"]["participants"]

    jungler_df = extract_jungler_data(timeline, participants)

    # Print the DataFrame to console
    print(jungler_df)

    # Optionally, save to CSV if you want:
    # jungler_df.to_csv("jungler_stats.csv", index=False)


if __name__ == "__main__":
    main()
