from pathlib import Path

import tensorflow as tf

from cancer_ml.model import get_simple_cnn

WEIGHTS_FILE = Path("/Users/mathis/Code/private_projects/cancer_ml/results/models/model.weights.h5")
DSET_FOLDER = Path("/Users/mathis/Code/private_projects/cancer_ml/results/datasets/train_100")
FILTER_SIZES = [32, 64, 128]
INPUT_SHAPE = (128, 256, 256, 1)

model = get_simple_cnn(INPUT_SHAPE, FILTER_SIZES)
model.load_weights(WEIGHTS_FILE)

# load data and sample info
print("---Load---")
ds = tf.data.Dataset.load(str(DSET_FOLDER))

print("---Convert---")
def convert_y(some_X: tf.Tensor, some_y: tf.Tensor) -> tuple:
    """
    We want y as float32 (but save as bool for smaller dsets)
    """
    some_X = tf.cast(some_X, tf.float16)
    some_y = tf.cast(some_y, tf.float16)
    return some_X, some_y


ds = ds.map(convert_y)
for X, y in ds.take(1):
    pass

print("---Predict---")
X = tf.expand_dims(X, axis=0)
y_pred = model.predict(x=X)
print(y_pred.shape)


