import requests
import time
import csv
from datetime import datetime

BASE_URL = "https://127.0.0.1:2999/liveclientdata"
OUTPUT_FILE = "enemy_pathing_data.csv"

# Disable warnings for self-signed certs from Riot's local HTTPS
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_json(endpoint):
    try:
        res = requests.get(f"{BASE_URL}/{endpoint}", verify=False)
        return res.json()
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        return {}


def get_visible_enemies():
    all_players = get_json("playerlist")

    # print(f"Found {len(all_players)} players in the game. {all_players}")
    active_player = get_json("activeplayer")
    active_team = active_player.get("team")

    visible_enemies = []

    for p in all_players:
        team = p.get("team")
        pos = p.get("position")
        is_visible = p.get("isVisible", False)

        if team and team != active_team and is_visible and isinstance(pos, dict):
            print(f"Found visible enemy: {p.get('summonerName')} at position {pos}")
            visible_enemies.append(
                {
                    "name": p.get("summonerName"),
                    "champion": p.get("championName"),
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "gold": p.get("currentGold", 0),
                    "cs": p.get("scores", {}).get("creepScore", 0),
                    "level": p.get("level", 0),
                }
            )

    return visible_enemies


def get_game_time():
    stats = get_json("gamestats")
    return stats.get("gameTime", 0.0)


def main():
    print("Tracking visible enemy data... (press Ctrl+C to stop)")

    with open(OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "game_time",
                "name",
                "champion",
                "x",
                "y",
                "gold",
                "cs",
                "level",
            ]
        )

        try:
            while True:
                game_time = get_game_time()
                enemies = get_visible_enemies()
                timestamp = datetime.utcnow().isoformat()

                for e in enemies:
                    writer.writerow(
                        [
                            timestamp,
                            f"{game_time:.2f}",
                            e["name"],
                            e["champion"],
                            e["x"],
                            e["y"],
                            e["gold"],
                            e["cs"],
                            e["level"],
                        ]
                    )
                    print(
                        f"[{game_time:.1f}s] {e['name']} ({e['champion']}): "
                        f"x={e['x']}, y={e['y']} | Gold={e['gold']} | CS={e['cs']} | Level={e['level']}"
                    )

                time.sleep(1.5)
        except KeyboardInterrupt:
            print("\nTracking stopped.")


if __name__ == "__main__":
    main()
