import os
from dotenv import load_dotenv

# DATA_VERSION = "v1"
# BASE_OUTPUT_DIR = os.path.join("database", "riot_data", DATA_VERSION)
# MATCH_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "match_data")
# SUMMONERS_CSV = os.path.join("database", "riot_data", "found_summoners.csv")
# MATCHES_PER_BATCH = 40


# load API key from .env
load_dotenv()  # Loads variables from .env into environment

RIOT_API_KEY = os.getenv("RIOT_API_KEY", "API Key not found")  # fallback if not set

REGION = "na1"
QUEUE_TYPE = "RANKED_SOLO_5x5"
# OUTPUT_CSV = "found_summoners.csv"

TIERS = ["DIAMOND", "PLATINUM"]
DIVISIONS = ["I", "II", "III", "IV"]

MAX_REQUESTS_PER_SECOND = 20
MAX_REQUESTS_PER_2_MINUTES = 100

# Paths
DATA_VERSION = "v2"
BASE_OUTPUT_DIR = os.path.join("database", "riot_data", DATA_VERSION)
MATCH_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "match_data")
SUMMONERS_CSV = os.path.join("database", "riot_data", "found_summoners.csv")

MATCHES_PER_BATCH = 40
