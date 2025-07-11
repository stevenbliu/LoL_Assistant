from RiotAPI import (
    get_summoner_data,
    get_ranked_match_ids,
    get_match_info,
    get_match_timeline,
)

import pandas as pd  # Import pandas


import pandas as pd
import math


def extract_jungler_data(timeline, participants):
    junglers = [p for p in participants if p["individualPosition"].upper() == "JUNGLE"]
    # Sort by participantId for consistency
    junglers = sorted(junglers, key=lambda p: p["participantId"])

    # Safety: if less than 2 junglers, return empty df or handle accordingly
    if len(junglers) < 2:
        print("Less than 2 junglers found in match.")
        return pd.DataFrame()

    # Take only the first two junglers
    j1, j2 = junglers[:2]

    # Collect timestamps (seconds)
    timestamps = []
    for frame in timeline["info"]["frames"]:
        timestamps.append(frame["timestamp"] // 1000)

    data = {
        "Second": timestamps,
        # Player 1 data
        "P1_Player": [],
        "P1_Champion": [],
        "P1_MinionsKilled": [],
        "P1_JungleMinionsKilled": [],
        "P1_X": [],
        "P1_Y": [],
        "P1_Team": [],
        "P1_Position": [],
        # Player 2 data
        "P2_Player": [],
        "P2_Champion": [],
        "P2_MinionsKilled": [],
        "P2_JungleMinionsKilled": [],
        "P2_X": [],
        "P2_Y": [],
        "P2_Team": [],
        "P2_Position": [],
        # Interaction features
        "Distance_Between_Junglers": [],
    }

    # Extract constant info for each player
    def get_player_info(j):
        pid = str(j["participantId"])
        return {
            "pid": pid,
            "player_name": j.get("summonerName")
            or j.get("riotIdGameName")
            or f"Player{pid}",
            "champion": j["championName"],
            "team": j["teamId"],
            "position": j["individualPosition"],
        }

    p1_info = get_player_info(j1)
    p2_info = get_player_info(j2)

    for frame in timeline["info"]["frames"]:
        p1_data = frame["participantFrames"][p1_info["pid"]]
        p2_data = frame["participantFrames"][p2_info["pid"]]

        # Positions
        p1_pos = p1_data.get("position", {"x": None, "y": None})
        p2_pos = p2_data.get("position", {"x": None, "y": None})

        # Calculate Euclidean distance if positions available
        if (
            p1_pos["x"] is not None
            and p1_pos["y"] is not None
            and p2_pos["x"] is not None
            and p2_pos["y"] is not None
        ):
            dist = math.sqrt(
                (p1_pos["x"] - p2_pos["x"]) ** 2 + (p1_pos["y"] - p2_pos["y"]) ** 2
            )
        else:
            dist = None

        # Append data
        data["P1_Player"].append(p1_info["player_name"])
        data["P1_Champion"].append(p1_info["champion"])
        data["P1_MinionsKilled"].append(p1_data.get("minionsKilled", 0))
        data["P1_JungleMinionsKilled"].append(p1_data.get("jungleMinionsKilled", 0))
        data["P1_X"].append(p1_pos.get("x"))
        data["P1_Y"].append(p1_pos.get("y"))
        data["P1_Team"].append(p1_info["team"])
        data["P1_Position"].append(p1_info["position"])

        data["P2_Player"].append(p2_info["player_name"])
        data["P2_Champion"].append(p2_info["champion"])
        data["P2_MinionsKilled"].append(p2_data.get("minionsKilled", 0))
        data["P2_JungleMinionsKilled"].append(p2_data.get("jungleMinionsKilled", 0))
        data["P2_X"].append(p2_pos.get("x"))
        data["P2_Y"].append(p2_pos.get("y"))
        data["P2_Team"].append(p2_info["team"])
        data["P2_Position"].append(p2_info["position"])

        data["Distance_Between_Junglers"].append(dist)

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


# def main():
# summoner_name = "Zdev#1111"
# game_name, tag_line = summoner_name.split("#")
# # game_name, tag_line = "Zdev", "1111"

# print(f"Fetching data for summoner: {summoner_name}")
# summoner_data = get_summoner_data(game_name, tag_line)
# puuid = summoner_data["puuid"]

# match_ids = get_recent_match_ids(puuid)
# match_id = match_ids[0]
# match_info = get_match_info(match_id)
# timeline = get_match_timeline(match_id)

# unique_events = set()
# for frame in timeline["info"]["frames"]:
#     if "events" in frame:
#         for event in frame["events"]:
#             if (
#                 event["type"] == "MONSTER_KILL"
#                 or event["type"] == "ELITE_MONSTER_KILL"
#             ):
#                 print(f"Timestamp: {event['timestamp']}ms")
#                 print(f"Monster Type: {event.get('monsterType', 'N/A')}")
#                 print(f"Killer Participant ID: {event.get('killerId', 'N/A')}")
#                 print(f"Monster Subtype: {event.get('monsterSubType', 'N/A')}")
#                 print("---")
#             # if event not in unique_events:
#             unique_events.add(event["type"])
# print(f"Unique events count: {unique_events}")

# # print_keys(match_info)
# # print_keys(timeline)

# participants = match_info["info"]["participants"]

# jungler_df = extract_jungler_data(timeline, participants)

# # Print the DataFrame to console
# print(jungler_df)

# Optionally, save to CSV if you want:
# jungler_df.to_csv("jungler_stats.csv", index=False)


# if __name__ == "__main__":
# main()
