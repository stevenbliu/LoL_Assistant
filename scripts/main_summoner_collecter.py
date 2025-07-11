import pandas as pd
import sys
import os

# Dynamically add project root (LoL_Assistant) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from riot.fetchers.fetch_summoners import find_good_summoners
from riot.config.config import SUMMONERS_CSV


def main():
    summoners = find_good_summoners()
    df = pd.DataFrame(summoners)
    df.to_csv(SUMMONERS_CSV, index=False)
    print(f"\n📝 Saved {len(df)} summoners to {SUMMONERS_CSV}")


if __name__ == "__main__":
    main()
