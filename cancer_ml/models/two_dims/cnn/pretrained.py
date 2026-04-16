"""
Use pretrained models.
"""
import keras
import keras_hub
from keras import layers


def get_pretrained_resnet() -> keras.Model:
    """
    Get ResNet 50 backbone.
    ResNets scale down by factor 32 and have 2048 filters at the end.

    Not ideal to use because you need to manually add a decoder on top.
    """
    backbone = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_50_imagenet"
    )
    return backbone


def get_pretrained_deeplab(num_classes: int = 1) -> keras.Model:
    """
    Get a pretrained DeepLabv3+ model.
    This is segmentation-ready.
    """
    model = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        preset="deeplab_v3_plus_resnet50_pascalvoc",
        num_classes=num_classes,
        activation="sigmoid"
    )
    return model


def unfreeze_last(model: keras.Model) -> keras.Model:
    """
    Unfreeze last layer of deeplab model.
    """
    model.trainable = True
    model.backbone.trainable = False  # let's not modify the resnet backbone
    return model


def unfreeze_aspp_decoder(model, also_batch_norm: bool = True) -> keras.Model:
    """
    Unfreeze everything except the pretrained encoder.
    """
    model.trainable = True
    encoder = model.get_layer("deep_lab_v3_backbone")
    for this_layer in encoder.layers:
        trainable = True
        if this_layer.name in ["inputs", "functional", "decoder_conv"]:
            trainable = False
        elif isinstance(this_layer, layers.BatchNormalization) and not also_batch_norm:
            trainable = False
        this_layer.trainable = trainable
        print(f"\t Layer {this_layer.name}: {trainable=}")
    return model