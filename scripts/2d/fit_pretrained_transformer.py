"""
Fit a pretrained vision transformer on the cancer segmentation task.
"""
from pathlib import Path

from cancer_ml.models.two_dims.transformer.pretrained import get_pretrained_model

# params
SOURCE = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d/samples500_uint8_val15_test15_512-512")

# load data


# go!
model = get_pretrained_model()

