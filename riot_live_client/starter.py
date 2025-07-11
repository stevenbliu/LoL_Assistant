import requests
import time
import json
import urllib3

# Disable SSL warnings from urllib3 when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://127.0.0.1:2999/liveclientdata"  # Note https://


def get_all_players():
    try:
        response = requests.get(f"{BASE_URL}/playerlist", verify=False, timeout=2)
        return response.json()
    except Exception as e:
        print(f"Error fetching player list: {e}")
        return []


def get_active_player():
    try:
        response = requests.get(f"{BASE_URL}/activeplayer", verify=False, timeout=2)
        return response.json()
    except Exception as e:
        print(f"Error fetching active player: {e}")
        return {}


def get_event_data():
    try:
        response = requests.get(f"{BASE_URL}/eventdata", verify=False, timeout=2)
        return response.json()
    except Exception as e:
        print(f"Error fetching events: {e}")
        return {}


def get_game_stats():
    try:
        response = requests.get(f"{BASE_URL}/gamestats", verify=False, timeout=2)
        return response.json()
    except Exception as e:
        print(f"Error fetching game stats: {e}")
        return {}


def get_active_player_team():
    active = get_active_player()
    return active.get("team")


def main_loop(poll_interval=1.0):
    print("Starting live client monitor...")
    while True:
        players = get_all_players()
        events = get_event_data()
        active = get_active_player()
        game = get_game_stats()

        print(f"\nGame Time: {game.get('gameTime', 0):.1f}s")
        print(
            f"Your Champion: {active.get('championName')} | Level: {active.get('level')} | Gold: {active.get('currentGold')}"
        )

        active_team = get_active_player_team()

        print("Visible Players:")
        for p in players:
            player_team = p.get("team")
            is_enemy = (
                (player_team != active_team) if (active_team and player_team) else False
            )

            pos = p.get("position")
            if isinstance(pos, dict):
                x = pos.get("x")
                y = pos.get("y")
            else:
                x = y = None

            is_visible = p.get("isVisible", False)

            print(
                f"{'[Enemy]' if is_enemy else '[Ally]'} {p.get('summonerName')} ({p.get('championName')}) - Visible: {is_visible} - Position: ({x}, {y})"
            )

        time.sleep(poll_interval)


if __name__ == "__main__":
    main_loop(3)
