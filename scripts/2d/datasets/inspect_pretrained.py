"""
Pretrained models often come with backed-in resizing and rescaling.
Let's find out what's going on
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cancer_ml.models.two_dims.pretrained import get_pretrained_deeplab

# params
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/2d/samples500_val15_test15_128-128")

# go!
i_sample = np.random.choice(100)


def print_stats(some_X) -> None:
    print(f"Shape: {some_X.shape}")
    print(f"Min: {some_X.min():.3f}")
    print(f"Mean: {some_X.mean():.3f}")
    print(f"Median: {np.median(some_X):.3f}")
    print(f"Max: {some_X.max():.3f}")


def recursive_layers(x, depth=0) -> None:
    if hasattr(x, "layers"):
        prefix = "\t" * depth
        print(f"{prefix} entering '{x.name}'")
        for l in x.layers:
            prefix = "\t" * (depth + 1)
            print(f"{prefix} {l}")
            recursive_layers(l, depth + 1)


train_ds = tf.data.Dataset.load(str(DSET_FOLDER / "train"))
X, y = next(iter(train_ds.skip(i_sample).take(1)))
print(X.shape)
print(y.shape)

print("---Model description---")
model = get_pretrained_deeplab()
recursive_layers(model)

print(f"---Pre---")
print_stats(X.numpy())


print("Scale:", model.preprocessor.image_converter.scale)
print("Offset:", model.preprocessor.image_converter.offset)
X_post = model.preprocessor(X)

print(f"---Post preprocessing---")
print_stats(X_post.numpy())


