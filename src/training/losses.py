import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MASKED BCE LOSS
# ============================================================

class MaskedBCELoss(nn.Module):
    """
    Binary Cross Entropy loss that ignores invalid pixels.

    predictions:
        Raw model logits
        [B, 1, H, W]

    targets:
        Ground truth masks
        [B, 1, H, W]

    valid_mask:
        1 = valid pixel
        0 = invalid pixel
        [B, 1, H, W]
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        predictions,
        targets,
        valid_mask
    ):

        # ----------------------------------------------------
        # Calculate BCE for every pixel
        # ----------------------------------------------------

        loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction="none"
        )

        # ----------------------------------------------------
        # Keep only valid pixels
        # ----------------------------------------------------

        valid_mask = valid_mask.float()

        loss = loss * valid_mask

        # ----------------------------------------------------
        # Avoid division by zero
        # ----------------------------------------------------

        valid_pixels = (
            valid_mask.sum()
        )

        if valid_pixels == 0:

            return torch.tensor(
                0.0,
                device=predictions.device,
                requires_grad=True
            )

        return (
            loss.sum()
            /
            valid_pixels
        )


# ============================================================
# MASKED DICE LOSS
# ============================================================

class MaskedDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Invalid pixels are ignored.
    """

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        predictions,
        targets,
        valid_mask
    ):

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        predictions = torch.sigmoid(
            predictions
        )

        targets = targets.float()
        valid_mask = valid_mask.float()

        # ----------------------------------------------------
        # Ignore invalid pixels
        # ----------------------------------------------------

        predictions = (
            predictions
            * valid_mask
        )

        targets = (
            targets
            * valid_mask
        )

        # ----------------------------------------------------
        # Flatten each sample
        # ----------------------------------------------------

        predictions = predictions.flatten(
            start_dim=1
        )

        targets = targets.flatten(
            start_dim=1
        )

        # ----------------------------------------------------
        # Dice components
        # ----------------------------------------------------

        intersection = (
            predictions
            * targets
        ).sum(dim=1)

        prediction_sum = (
            predictions.sum(dim=1)
        )

        target_sum = (
            targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            prediction_sum
            + target_sum
            + self.smooth
        )

        # ----------------------------------------------------
        # Dice Loss
        # ----------------------------------------------------

        return (
            1.0 - dice
        ).mean()


# ============================================================
# COMBINED BCE + DICE LOSS
# ============================================================

class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    Total loss:

        loss =
            bce_weight * BCE
            +
            dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):

        super().__init__()

        self.bce_weight = (
            bce_weight
        )

        self.dice_weight = (
            dice_weight
        )

        self.bce = MaskedBCELoss()

        self.dice = MaskedDiceLoss()

    def forward(
        self,
        predictions,
        targets,
        valid_mask
    ):

        bce_loss = self.bce(
            predictions,
            targets,
            valid_mask
        )

        dice_loss = self.dice(
            predictions,
            targets,
            valid_mask
        )

        total_loss = (
            self.bce_weight
            * bce_loss
            +
            self.dice_weight
            * dice_loss
        )

        return total_loss


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LOSS FUNCTION TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Fake model predictions
    # --------------------------------------------------------

    predictions = torch.randn(
        2,
        1,
        512,
        512,
        requires_grad=True
    )

    # --------------------------------------------------------
    # Fake ground truth
    # --------------------------------------------------------

    targets = torch.zeros(
        2,
        1,
        512,
        512
    )

    # Add some artificial oil regions
    targets[
        :,
        :,
        200:300,
        200:300
    ] = 1.0

    # --------------------------------------------------------
    # Valid mask
    # --------------------------------------------------------

    valid_mask = torch.ones(
        2,
        1,
        512,
        512
    )

    # Make a small invalid region
    valid_mask[
        :,
        :,
        0:50,
        0:50
    ] = 0.0

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = BCEDiceLoss()

    loss = criterion(
        predictions,
        targets,
        valid_mask
    )

    print(
        f"Prediction shape: "
        f"{predictions.shape}"
    )

    print(
        f"Target shape: "
        f"{targets.shape}"
    )

    print(
        f"Valid mask shape: "
        f"{valid_mask.shape}"
    )

    print(
        f"Loss: "
        f"{loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Backpropagation test
    # --------------------------------------------------------

    loss.backward()

    print(
        "Gradient calculated: "
        f"{predictions.grad is not None}"
    )

    print("=" * 60)

    if (
        torch.isfinite(loss)
        and
        predictions.grad is not None
    ):

        print(
            "✅ Loss function test passed"
        )

    else:

        print(
            "❌ Loss function test failed"
        )