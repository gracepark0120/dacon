import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SMP(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
configs = {
    "model": {
        "encoder_name": "timm-regnety_320",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 1
    },
    "data": {
        "root": ".",
        "batch_size": 64
    },
}

model = SMP(
            encoder_name=configs["model"]["encoder_name"],
            encoder_weights=configs["model"]["encoder_weights"],
            in_channels=configs["model"]["in_channels"],
            classes=configs["model"]["classes"],
        )

model = model.to(device)