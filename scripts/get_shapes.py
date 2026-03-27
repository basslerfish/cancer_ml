"""
The T1 and GTV data shapes are apparently not standardized.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from cancer_ml.load import find_and_load_sample

# paths
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/data/BraTS-MEN-RT-Train-v2")
OUTPUT = Path("/Users/mathis/Code/private_projects/cancer_ml/results")

# go
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
            "min_val": np.min(t1),
            "max_val": np.max(t1),
            "median_val": np.median(t1),
            "mean_val": np.mean(t1),
        }
        entry = pd.DataFrame([entry])
        if count == 0:
            entry.to_csv(save_file)
        else:
            entry.to_csv(save_file, mode="a", header=False)
        count += 1