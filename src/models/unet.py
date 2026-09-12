import torch
import torch.nn as nn


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        input_channels=2,
        output_channels=1,
        base_features=64
    ):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder1 = DoubleConv(
            input_channels,
            base_features
        )

        self.encoder2 = DoubleConv(
            base_features,
            base_features * 2
        )

        self.encoder3 = DoubleConv(
            base_features * 2,
            base_features * 4
        )

        self.encoder4 = DoubleConv(
            base_features * 4,
            base_features * 8
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            base_features * 8,
            base_features * 16
        )

        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.upconv4 = nn.ConvTranspose2d(
            base_features * 16,
            base_features * 8,
            kernel_size=2,
            stride=2
        )

        self.decoder4 = DoubleConv(
            base_features * 16,
            base_features * 8
        )

        self.upconv3 = nn.ConvTranspose2d(
            base_features * 8,
            base_features * 4,
            kernel_size=2,
            stride=2
        )

        self.decoder3 = DoubleConv(
            base_features * 8,
            base_features * 4
        )

        self.upconv2 = nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(
            base_features * 4,
            base_features * 2
        )

        self.upconv1 = nn.ConvTranspose2d(
            base_features * 2,
            base_features,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(
            base_features * 2,
            base_features
        )

        # ----------------------------------------------------
        # Final segmentation layer
        # ----------------------------------------------------

        self.final_conv = nn.Conv2d(
            base_features,
            output_channels,
            kernel_size=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(
            self.pool(enc1)
        )

        enc3 = self.encoder3(
            self.pool(enc2)
        )

        enc4 = self.encoder4(
            self.pool(enc3)
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        bottleneck = self.bottleneck(
            self.pool(enc4)
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        dec4 = self.upconv4(
            bottleneck
        )

        dec4 = torch.cat(
            [
                dec4,
                enc4
            ],
            dim=1
        )

        dec4 = self.decoder4(
            dec4
        )

        # ----------------------------------------------------

        dec3 = self.upconv3(
            dec4
        )

        dec3 = torch.cat(
            [
                dec3,
                enc3
            ],
            dim=1
        )

        dec3 = self.decoder3(
            dec3
        )

        # ----------------------------------------------------

        dec2 = self.upconv2(
            dec3
        )

        dec2 = torch.cat(
            [
                dec2,
                enc2
            ],
            dim=1
        )

        dec2 = self.decoder2(
            dec2
        )

        # ----------------------------------------------------

        dec1 = self.upconv1(
            dec2
        )

        dec1 = torch.cat(
            [
                dec1,
                enc1
            ],
            dim=1
        )

        dec1 = self.decoder1(
            dec1
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        return self.final_conv(
            dec1
        )


# ============================================================
# QUICK MODEL TEST
# ============================================================

if __name__ == "__main__":

    model = UNet(
        input_channels=2,
        output_channels=1,
        base_features=64
    )

    dummy_input = torch.randn(
        2,
        2,
        512,
        512
    )

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print("=" * 60)
    print("U-NET TEST")
    print("=" * 60)

    print(
        f"Input shape:  "
        f"{dummy_input.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print("=" * 60)

    if output.shape == (
        2,
        1,
        512,
        512
    ):

        print(
            "✅ U-Net test passed"
        )

    else:

        print(
            "❌ U-Net test failed"
        )