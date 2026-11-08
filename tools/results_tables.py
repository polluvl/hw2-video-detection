"""results.csv -> markdown tables (one per detector family) to paste into the report.
Usage: python tools/results_tables.py path/to/results.csv
"""
import sys

import pandas as pd

COLS = ["experiment", "mAP", "AP50", "AP_small", "AP_medium", "AP_large", "ms_per_img", "train_min", "config"]


def main(path):
    df = pd.read_csv(path)
    for family, group in df.groupby("family", sort=False):
        print(f"\n### {family}\n")
        print(group[[c for c in COLS if c in group]].to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/results.csv")
