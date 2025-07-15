from training.config import DATA_DIR, VERSION
import pandas as pd


raw_df_path = f"{DATA_DIR}/{VERSION}/all_jungler_data.csv"
raw_df = pd.read_csv(raw_df_path)


cache_path = (
    f"I:\GitHub\LoL_Assistant\cache\seq_c5a108cb228fdb5867e817c1bfe74330.feather"
)
sequence_df = pd.read_feather(cache_path)

print(raw_df.head(3))

print(sequence_df.head(3))
