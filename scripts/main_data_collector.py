import os
import sys

# Dynamically add project root (LoL_Assistant) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from riot.fetchers.collect_matches import collect_data_from_summoners
from riot.processing.utils import save_combined_dataset
from riot.config.config import BASE_OUTPUT_DIR


def main():
    all_jungler_dfs = collect_data_from_summoners()
    save_combined_dataset(all_jungler_dfs, BASE_OUTPUT_DIR)


if __name__ == "__main__":
    main()
