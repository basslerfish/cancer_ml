"""
Pretrained vision transformer (ViT) segmentation models.
"""
import keras
import keras_hub


def unfreeze_final(model: keras.Model) -> keras.Model:
    model.trainable = True
    for layer in model.layers:
        if layer.name == "conv2d_11":
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