import json
import pandas as pd
from pathlib import Path
from collections import Counter

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATES_DIR = BASE_DIR / "states"

STATE_FILES = [
    "massachusetts.json",
    "california.json",
    "new_york.json",
    "texas.json",
    "florida.json",
]

OUT_JSON = BASE_DIR / "traffic_laws_dataset.json"
OUT_CSV = BASE_DIR / "traffic_laws_dataset.csv"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
REQUIRED_KEYS = {
    "id",
    "jurisdiction",
    "category",
    "violation",
    "law_text",
    "statute",
    "penalty",
    "severity",
    "preventive_tip",
    "keywords",
}

def validate_record(record: dict, source: str):
    missing = REQUIRED_KEYS - set(record.keys())
    if missing:
        raise ValueError(
            f"Missing keys {missing} in record id={record.get('id')} from {source}"
        )

    if not isinstance(record.get("keywords"), list):
        raise ValueError(
            f"'keywords' must be a list in record id={record.get('id')} from {source}"
        )

def load_state_file(fp: Path):
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{fp.name} must contain a list of records")

    for r in data:
        validate_record(r, fp.name)

    return data

# -----------------------------------------------------------------------------
# Main build
# -----------------------------------------------------------------------------
def main():
    if not STATES_DIR.exists():
        raise FileNotFoundError(f"States folder not found at: {STATES_DIR}")

    traffic_laws_data = []

    for filename in STATE_FILES:
        fp = STATES_DIR / filename
        if not fp.exists():
            raise FileNotFoundError(f"Missing state file: {fp}")

        state_records = load_state_file(fp)
        traffic_laws_data.extend(state_records)

    # Check duplicate IDs efficiently
    ids = [r["id"] for r in traffic_laws_data]
    id_counts = Counter(ids)
    dupes = [i for i, c in id_counts.items() if c > 1]
    if dupes:
        raise ValueError(
            f"Duplicate IDs found across state files: {sorted(dupes)}\n"
            "Please ensure IDs are unique per record."
        )

    df = pd.DataFrame(traffic_laws_data)

    df.to_csv(OUT_CSV, index=False)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(traffic_laws_data, f, indent=2, ensure_ascii=False)

    print("Dataset created successfully!")
    print(f"JSON saved to: {OUT_JSON}")
    print(f"CSV saved to:  {OUT_CSV}")

    print(f"\nDataset Statistics:")
    print(f"Total records: {len(traffic_laws_data)}")
    print(f"Jurisdictions: {df['jurisdiction'].nunique()}")
    print(f"Categories: {sorted(df['category'].unique().tolist())}")

    print("\nSample record:")
    print(json.dumps(traffic_laws_data[0], indent=2))

if __name__ == "__main__":
    main()