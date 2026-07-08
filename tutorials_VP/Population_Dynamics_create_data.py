#!/usr/bin/env python
# coding: utf-8
"""Create normalized training, validation, and mock data for Population_Dynamics.py."""
from __future__ import annotations

import argparse

from autocvd import autocvd

from Population_Dynamics import POPULATION_DATA_PATH, create_population_data


def main() -> None:
    autocvd(num_gpus=1, interval=1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all training, validation, and confusion data; trained models must then be retrained.",
    )
    args = parser.parse_args()
    data = create_population_data(force=args.force)
    print(f"Ready data file: {POPULATION_DATA_PATH}")
    print(f"Stored models: {data['model_names']}")


if __name__ == "__main__":
    main()
