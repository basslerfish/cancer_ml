import keras
import keras_hub


def get_pretrained_deeplab(num_classes: int = 1) -> keras.Model:
    model = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        preset="deeplab_v3_plus_resnet50_pascalvoc",
        num_classes=num_classes,
        activation="sigmoid"
    )
    return model


def get_pretrained_resnet() -> keras.Model:
    """
    Get ResNet 50 backbone
    ResNets scale down by factor 32 and have 2048 filters at the end.

    Not ideal to use because you need to manually add a decoder on top.
    """
    backbone = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_50_imagenet"
    )
    return backbone