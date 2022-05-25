import segmentation_models_pytorch as smp


def get_model():
    # TODO TIP: There's a lot of tasty things to try here.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model
