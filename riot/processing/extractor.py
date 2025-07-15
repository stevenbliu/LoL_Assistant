import pandas as pd  # Import pandas
import math
from collections import defaultdict


from collections import defaultdict
import pandas as pd
import math


def extract_jungler_data(timeline, participants):
    junglers = [p for p in participants if p["individualPosition"].upper() == "JUNGLE"]
    junglers = sorted(junglers, key=lambda p: p["participantId"])

    if len(junglers) < 2:
        print("❌ Less than 2 junglers found.")
        return pd.DataFrame()

    j1, j2 = junglers[:2]

    def get_player_info(p):
        pid = str(p["participantId"])
        return {
            "participantId": pid,
            "playerId": p["puuid"],
            "player_name": p.get("summonerName")
            or p.get("riotIdGameName")
            or f"Player{pid}",
            "champion": p["championName"],
            "team": p["teamId"],
            "position": p["individualPosition"],
        }

    participant_info = {
        str(p["participantId"]): get_player_info(p) for p in participants
    }

    p1_id, p2_id = j1["participantId"], j2["participantId"]
    data = defaultdict(list)

    for frame in timeline["info"]["frames"]:
        pf = frame.get("participantFrames", {})
        timestamp = frame["timestamp"] // 1000
        data["Timestamp"].append(timestamp)

        for pid_str in participant_info:
            info = participant_info[pid_str]
            stats = pf.get(pid_str)  # This may be None

            prefix = (
                "P1"
                if pid_str == str(p1_id)
                else ("P2" if pid_str == str(p2_id) else f"NP{pid_str}")
            )

            # Always append timestamp for every player
            data[f"{prefix}_Player"].append(info["player_name"])
            data[f"{prefix}_PlayerId"].append(info["playerId"])
            data[f"{prefix}_Participant"].append(info["participantId"])
            data[f"{prefix}_Champion"].append(info["champion"])
            data[f"{prefix}_Team"].append(info["team"])
            data[f"{prefix}_Position"].append(info["position"])

            if stats:
                data[f"{prefix}_MinionsKilled"].append(stats.get("minionsKilled", 0))
                data[f"{prefix}_JungleMinionsKilled"].append(
                    stats.get("jungleMinionsKilled", 0)
                )
                data[f"{prefix}_X"].append(stats.get("position", {}).get("x", math.nan))
                data[f"{prefix}_Y"].append(stats.get("position", {}).get("y", math.nan))
                data[f"{prefix}_currentGold"].append(stats.get("currentGold", 0))
                data[f"{prefix}_level"].append(stats.get("level", 0))
                data[f"{prefix}_xp"].append(stats.get("xp", 0))
                data[f"{prefix}_goldPerSecond"].append(stats.get("goldPerSecond", 0))

                for stat_name in stats.get("damageStats", {}):
                    data[f"{prefix}_{stat_name}"].append(
                        stats["damageStats"].get(stat_name, 0)
                    )

                for stat_name in stats.get("championStats", {}):
                    data[f"{prefix}_{stat_name}"].append(
                        stats["championStats"].get(stat_name, 0)
                    )
            else:
                # Fill with None/NaN when stats are missing
                data[f"{prefix}_MinionsKilled"].append(0)
                data[f"{prefix}_JungleMinionsKilled"].append(0)
                data[f"{prefix}_X"].append(math.nan)
                data[f"{prefix}_Y"].append(math.nan)
                data[f"{prefix}_currentGold"].append(0)
                data[f"{prefix}_level"].append(0)
                data[f"{prefix}_xp"].append(0)
                data[f"{prefix}_goldPerSecond"].append(0)

                # Fill in known stat keys with zeros or NaN
                # Optional: define expected stat keys explicitly
                for stat in [
                    "physicalVamp",
                    "power",
                    "powerMax",
                    "powerRegen",
                    "spellVamp",
                ]:
                    data[f"{prefix}_{stat}"].append(0)

    # Sanity check
    lengths = {k: len(v) for k, v in data.items()}
    if len(set(lengths.values())) != 1:
        print("❌ Column length mismatch detected:")
        for k, v in sorted(lengths.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        raise ValueError("All arrays must be of the same length")

    return pd.DataFrame(data)


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
