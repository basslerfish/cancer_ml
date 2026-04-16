"""
The T1 and GTV data shapes are apparently not standardized.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cancer_ml.load import find_and_load_sample

# paths
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
TABLE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/extra_info.xlsx")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results")
COLS_TO_EXTRACT = {
    "T1c x-resolution (mm)": "x_res",
    "T1c y-resolution (mm)": "y_res",
    "T1c Slice Thickness (mm)": "z_res",
}

# go
extra_info = pd.read_excel(TABLE, sheet_name="BraTS-MEN-RT Clinical Data")
save_file = OUTPUT / "train_samples.csv"
count = 0
for element in SOURCE.iterdir():
    print(f"{element.name}")
    if element.is_dir():
        t1, gtv = find_and_load_sample(element)
        entry = {
            "sample": element.name,
            "x": t1.shape[0],
            "y": t1.shape[1],
            "z": t1.shape[2],
            "aspect": t1.shape[0] / t1.shape[1],
            "min_val": np.min(t1),
            "max_val": np.max(t1),
            "median_val": np.median(t1),
            "mean_val": np.mean(t1),
        }
        # extra
        is_row = extra_info["BraTS ID"] == element.name
        assert np.sum(is_row) == 1
        for col, new_name in COLS_TO_EXTRACT.items():
            entry[new_name] = extra_info.loc[is_row, col].values[0]

        entry = pd.DataFrame([entry])
        if count == 0:
            entry.to_csv(save_file)
        else:
            entry.to_csv(save_file, mode="a", header=False)
        count += 1