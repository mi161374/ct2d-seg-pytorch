# src/models/unet.py
from monai.networks.nets import UNet

def build_unet(in_channels=1, out_channels=8):
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
