import json
from datetime import datetime
import os
from riot.config.config import DATA_VERSION


def save_metadata(output_dir, dataframe):
    metadata = {
        "version": DATA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_rows": len(dataframe),
        "num_matches": dataframe["MatchId"].nunique(),
        "features": [
            col
            for col in dataframe.columns
            if col
            not in ["P1_x_next", "P1_y_next", "P2_x_next", "P2_y_next", "MatchId"]
        ],
        "labels": ["P1_x_next", "P1_y_next", "P2_x_next", "P2_y_next"],
        "match_id_column": "MatchId",
        "source": "League of Legends Ranked Solo 5v5 via Riot API",
    }

    meta_path = os.path.join(output_dir, "meta_data.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"📝 Metadata saved to: {meta_path}")
