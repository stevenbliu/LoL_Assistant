import requests
from datetime import datetime
import time
import urllib3

# Disable insecure HTTPS warnings for localhost
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SelfJungleTracker:
    def __init__(self):
        self.history = []

    def fetch_active_player(self):
        try:
            response = requests.get(
                "https://127.0.0.1:2999/liveclientdata/activeplayer", verify=False
            )
            return response.json()
        except Exception as e:
            print(f"[ERROR] Fetching active player failed: {e}")
            return None

    def fetch_active_scores(self, riotId):
        try:
            response = requests.get(
                f"https://127.0.0.1:2999/liveclientdata/playerscores?riotId={riotId}",
                verify=False,
            )
            return response.json()
        except Exception as e:
            print(f"[ERROR] Fetching active player scores failed: {e}")
            return None

    def fetch_active_items(self):
        try:
            response = requests.get(
                "https://127.0.0.1:2999/liveclientdata/playeritems?riotId=",
                verify=False,
            )
            return response.json()
        except Exception as e:
            print(f"[ERROR] Fetching active player items failed: {e}")
            return []

    def update_info(self):
        player = self.fetch_active_player()
        riotId = player.get("riotId") if player else None
        scores = self.fetch_active_scores(riotId)
        items = self.fetch_active_items()

        print("Scores:", scores)

        if player and scores is not None:
            entry = {
                "name": player.get("championName", "Unknown"),
                "level": player.get("level", 0),
                "cs": scores.get("creepScore", 0),
                "isDead": player.get("isDead", False),
                # "items": [item["displayName"] for item in items],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "riotId": riotId,
            }
            self.history.append(entry)
            return entry
        return None

    def infer_pathing(self, data):
        cs = data["cs"]
        level = data["level"]
        isDead = data["isDead"]
        time_seen = data["timestamp"]
        inference = f"{data['name']} (Lv {level}, {cs} CS) at {time_seen}: "

        if cs < 4:
            inference += "Likely did 1 camp or still starting."
        elif cs <= 8:
            inference += "Probably cleared 2 camps."
        elif cs <= 12:
            inference += "3 camps cleared, possible scuttle soon."
        elif cs <= 20:
            inference += "Likely 4–5 camps cleared."
        else:
            inference += "Likely full clear + recall."

        if isDead:
            inference += " (Currently dead)"

        inference += f" CS: {cs} RiotId: {data['riotId']}"

        return inference


def main_loop(poll_interval=5):
    tracker = SelfJungleTracker()
    print("Tracking your jungle data...")

    while True:
        data = tracker.update_info()
        if data:
            print("\n--- Jungle Pathing Inference ---")
            print(tracker.infer_pathing(data))
        else:
            print("Waiting for live game data...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main_loop()
