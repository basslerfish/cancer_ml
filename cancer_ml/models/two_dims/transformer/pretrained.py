"""
Pretrained vision transformer (ViT) segmentation models.
"""
import keras
import keras_hub


def unfreeze_final(model: keras.Model) -> keras.Model:
    """Unfreeze the final layer of a size 0 segformer model.

    Frozen parts:
    0  input_layer_1 (trainable=False, params=0)
    0  seg_former_backbone (trainable=False, params=3,714,656)
    0  dropout_16 (trainable=False, params=0)

    Unfrozen parts (really last conv layer only):
    0  conv2d_11 (trainable=True, params=257)
    0  resizing_1 (trainable=False, params=0)
    """
    model.trainable = True
    for layer in model.layers:
        if layer.name == "conv2d_11":
            layer.trainable = True
        else:
            layer.trainable = False
    return model


def unfreeze_post_encoder(model: keras.Model) -> keras.Model:
    """
    Strictly speaking, we are unfreezing somewhere within the encoder.

    Frozen parts:
    0  input_layer_1 (trainable=False, params=0)
    0  seg_former_backbone (trainable=True, params=3,714,656)
    1 	 input_layer_1 (trainable=False, params=0)

    Unfrozen parts:
    1 	 functional (trainable=False, params=3,319,392)
    1 	 linear_256 (trainable=True, params=65,792)
    1 	 linear_160 (trainable=True, params=41,216)
    1 	 linear_64 (trainable=True, params=16,640)
    1 	 linear_32 (trainable=True, params=8,448)
    1 	 resizing (trainable=False, params=0)
    1 	 concatenate (trainable=False, params=0)
    1 	 sequential (trainable=True, params=263,168)
    2 		 conv2d_10 (trainable=True, params=262,144)
    2 		 batch_normalization (trainable=True, params=1,024)
    2 		 activation (trainable=True, params=0)
    0  dropout_16 (trainable=True, params=0)
    0  conv2d_11 (trainable=True, params=257)
    0  resizing_1 (trainable=True, params=0)
    """
    model.trainable = True
    for layer in model.backbone.layers:
        if (layer.name == "sequential") or ("linear" in layer.name):
            layer.trainable = True
        else:
            layer.trainable = False
    return model


def get_pretrained_model(
        size_index: int = 0,
        num_classes: int = 1,
) -> keras.Model:
    """
    Get a SegFormer ViT segmentation model from the 2021 paper.
    Note that the NN layer (a 1x1 conv layer to reduce filter_size to num_classes) has no activation.
    That means the loss will have to take that into account (from_logits=True).
    """
    segmenter = keras_hub.models.SegFormerImageSegmenter.from_preset(
        f"segformer_b{size_index}_ade20k_512",
        num_classes=num_classes,
    )
    return segmenter