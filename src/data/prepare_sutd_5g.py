"""Extract Time/RSRP from the four SUTD 5G scenarios and split each in half."""

import argparse
from pathlib import Path

import pandas as pd


SOURCE_FILES = [
    "Lvl4_AllRRUOn_Anomaly_label.csv",
    "Lvl5_AllRRUOn_Anomaly_label.csv",
    "Lvl6_1RRUOn_Anomaly_label.csv",
    "Lvl6_AllRRUOn_Anomaly_label.csv",
]


def prepare_sutd_5g(input_dir, output_dir):
    """Write the chronological first/second half of every scenario to train/evaluation."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    train_dir, evaluation_dir = output_dir / "train", output_dir / "evaluation"
    train_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    for filename in SOURCE_FILES:
        source_path = input_dir / filename
        source = pd.read_csv(source_path, usecols=["Time", "RSRP"])
        trace = source.rename(columns={"Time": "time", "RSRP": "rsrp"}).dropna()
        trace["time"] = pd.to_datetime(trace["time"])
        trace = trace.sort_values("time").reset_index(drop=True)
        midpoint = len(trace) // 2
        if midpoint == 0 or len(trace) - midpoint == 0:
            raise ValueError(f"Cannot split an empty scenario: {source_path}")

        trace.iloc[:midpoint].to_csv(train_dir / filename, index=False)
        trace.iloc[midpoint:].to_csv(evaluation_dir / filename, index=False)
        print(f"{filename}: train={midpoint}, evaluation={len(trace) - midpoint}")


def main():
    parser = argparse.ArgumentParser(description="Prepare chronological SUTD 5G train/evaluation traces.")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed/sutd_5g")
    args = parser.parse_args()
    prepare_sutd_5g(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
