"""
Oil Spill Detection - U-Net Training

Training configuration:
    - Input: VV + VH Sentinel-1 SAR channels
    - Patch size: 512x512
    - Model: U-Net
    - Loss: BCE + Dice
    - Optimizer: Adam
    - Batch size: 4
    - Workers: 6
    - AMP: enabled on CUDA
    - Maximum epochs: 50

First run:
    RUN_SANITY_ONLY = True

After verifying the 1-epoch result:
    RUN_SANITY_ONLY = False

The second run automatically resumes from latest_checkpoint.pth.
"""

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloader
from src.models.unet import UNet
from src.training.losses import BCEDiceLoss


# =====================================================================
# CONFIGURATION
# =====================================================================

# -------------------------
# Training
# -------------------------

EPOCHS = 50

# IMPORTANT:
# Keep True for the first run.
# Change to False after checking the 1-epoch results.
RUN_SANITY_ONLY = False

SANITY_EPOCHS = 1

BATCH_SIZE = 4
NUM_WORKERS = 6

LEARNING_RATE = 1e-4

# -------------------------
# Model
# -------------------------

INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 1
BASE_FEATURES = 64

# -------------------------
# Segmentation
# -------------------------

DICE_THRESHOLD = 0.5

# -------------------------
# Checkpointing
# -------------------------

MODEL_DIR = PROJECT_ROOT / "models"

BEST_MODEL_PATH = MODEL_DIR / "best_unet.pth"
LATEST_CHECKPOINT_PATH = MODEL_DIR / "latest_checkpoint.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"

RESUME_TRAINING = True

# -------------------------
# Scheduler
# -------------------------

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LEARNING_RATE = 1e-6

# -------------------------
# Early stopping
# -------------------------

EARLY_STOPPING_PATIENCE = 7
MIN_IMPROVEMENT = 1e-4

# -------------------------
# Reproducibility
# -------------------------

SEED = 42


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Do not force deterministic CUDA algorithms.
    # That can significantly reduce training speed.
    torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =====================================================================
# DEVICE
# =====================================================================

def get_device():
    """Select CUDA when available."""

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("=" * 70)
        print("CUDA AVAILABLE")
        print("=" * 70)
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version     : {torch.version.cuda}")
        print(f"GPU count        : {torch.cuda.device_count()}")

        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(
            f"GPU memory       : "
            f"{total_memory / (1024 ** 3):.2f} GB"
        )

        print("=" * 70)

        return device

    print("=" * 70)
    print("WARNING: CUDA NOT AVAILABLE")
    print("Training will run on CPU.")
    print("=" * 70)

    return torch.device("cpu")


# =====================================================================
# METRICS
# =====================================================================

class MetricAccumulator:
    """
    Dataset-level foreground segmentation metrics.

    We aggregate TP, FP and FN across all valid pixels.

    This avoids artificially high Dice scores caused by patches
    where both prediction and ground truth contain no oil.
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, predictions, targets, valid_mask):
        """
        predictions : [B, H, W]
        targets     : [B, H, W]
        valid_mask  : [B, H, W]
        """

        predictions = predictions.bool()
        targets = targets.bool()
        valid_mask = valid_mask.bool()

        tp = (
            predictions
            & targets
            & valid_mask
        ).sum().item()

        fp = (
            predictions
            & ~targets
            & valid_mask
        ).sum().item()

        fn = (
            ~predictions
            & targets
            & valid_mask
        ).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        """Calculate Dice, IoU, precision and recall."""

        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)

        dice_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn

        precision_denominator = tp + fp
        recall_denominator = tp + fn

        dice = (
            (2 * tp) / dice_denominator
            if dice_denominator > 0
            else 0.0
        )

        iou = (
            tp / iou_denominator
            if iou_denominator > 0
            else 0.0
        )

        precision = (
            tp / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        return {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        }


# =====================================================================
# TRAINING
# =====================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    total_epochs,
):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader, start=1):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        valid_masks = batch["valid_mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        # -------------------------------------------------------------
        # Mixed precision forward + loss
        # -------------------------------------------------------------

        with autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                masks,
                valid_masks,
            )

        # -------------------------------------------------------------
        # Backward
        # -------------------------------------------------------------

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # -------------------------------------------------------------
        # Loss accumulation
        # -------------------------------------------------------------

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # -------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------

        if batch_idx % 100 == 0 or batch_idx == len(loader):

            elapsed = time.time() - start_time

            batches_per_second = batch_idx / max(elapsed, 1e-6)

            remaining_batches = len(loader) - batch_idx

            eta_seconds = (
                remaining_batches / batches_per_second
                if batches_per_second > 0
                else 0
            )

            print(
                f"\r"
                f"Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.4f} "
                f"ETA: {eta_seconds / 60:.1f} min",
                end="",
                flush=True,
            )

    print()

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    return average_loss


# =====================================================================
# VALIDATION
# =====================================================================

def validate(
    model,
    loader,
    criterion,
    device,
):
    """Run validation."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    metrics = MetricAccumulator()

    start_time = time.time()

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader, start=1):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            valid_masks = batch["valid_mask"].to(
                device,
                non_blocking=True,
            )

            # ---------------------------------------------------------
            # Forward
            # ---------------------------------------------------------

            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    masks,
                    valid_masks,
                )

            # ---------------------------------------------------------
            # Loss
            # ---------------------------------------------------------

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # ---------------------------------------------------------
            # Predictions
            # ---------------------------------------------------------

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= DICE_THRESHOLD
            )

            targets = masks >= 0.5

            metrics.update(
                predictions,
                targets,
                valid_masks,
            )

            if batch_idx % 100 == 0 or batch_idx == len(loader):

                elapsed = time.time() - start_time

                print(
                    f"\rValidation "
                    f"Batch [{batch_idx}/{len(loader)}]",
                    end="",
                    flush=True,
                )

    print()

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    metric_values = metrics.compute()

    return average_loss, metric_values


# =====================================================================
# CHECKPOINT
# =====================================================================

def save_latest_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_dice,
    epochs_without_improvement,
    history,
):
    """Save complete checkpoint for resuming training."""

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "best_dice": best_dice,

        "epochs_without_improvement": (
            epochs_without_improvement
        ),

        "history": history,

        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "input_channels": INPUT_CHANNELS,
            "output_channels": OUTPUT_CHANNELS,
            "base_features": BASE_FEATURES,
            "dice_threshold": DICE_THRESHOLD,
        },
    }

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT_PATH,
    )


def save_best_model(model, epoch, dice):
    """
    Save only model weights.

    Keeping best_unet.pth as a state_dict makes it easy to use
    with existing evaluation/inference code.
    """

    torch.save(
        model.state_dict(),
        BEST_MODEL_PATH,
    )

    print()
    print("=" * 70)
    print("NEW BEST MODEL")
    print("=" * 70)
    print(f"Epoch : {epoch}")
    print(f"Dice  : {dice:.6f}")
    print(f"Saved : {BEST_MODEL_PATH}")
    print("=" * 70)


# =====================================================================
# RESUME
# =====================================================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    """Resume from the latest checkpoint if available."""

    if not LATEST_CHECKPOINT_PATH.exists():

        print("No previous checkpoint found.")
        print("Starting training from epoch 1.")

        return (
            1,
            -1.0,
            0,
            [],
        )

    print()
    print("=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)
    print(LATEST_CHECKPOINT_PATH)

    checkpoint = torch.load(
        LATEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    last_epoch = checkpoint["epoch"]

    best_dice = checkpoint.get(
        "best_dice",
        -1.0,
    )

    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0,
    )

    history = checkpoint.get(
        "history",
        [],
    )

    start_epoch = last_epoch + 1

    print(f"Resuming from epoch : {last_epoch}")
    print(f"Next epoch          : {start_epoch}")
    print(f"Best Dice           : {best_dice:.6f}")

    print("=" * 70)

    return (
        start_epoch,
        best_dice,
        epochs_without_improvement,
        history,
    )


# =====================================================================
# HISTORY
# =====================================================================

def save_history_csv(history):
    """Save training history as CSV."""

    if not history:
        return

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "dice",
        "iou",
        "precision",
        "recall",
        "learning_rate",
        "epoch_time_minutes",
    ]

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in history:
            writer.writerow(row)


# =====================================================================
# MAIN
# =====================================================================

def main():

    # -------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------

    set_seed(SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    if device.type != "cuda":

        print()
        print(
            "ERROR: CUDA GPU is required for this training run."
        )
        print(
            "Your tested configuration uses the RTX 3050."
        )
        print()

        return

    # -------------------------------------------------------------
    # Training length
    # -------------------------------------------------------------

    if RUN_SANITY_ONLY:
        target_epochs = SANITY_EPOCHS
    else:
        target_epochs = EPOCHS

    # -------------------------------------------------------------
    # Configuration display
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("OIL SPILL DETECTION - U-NET TRAINING")
    print("=" * 70)

    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Device             : {device}")
    print(f"Model              : U-Net")
    print(f"Input channels     : {INPUT_CHANNELS} (VV + VH)")
    print(f"Patch size         : 512 x 512")
    print(f"Base features      : {BASE_FEATURES}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Workers            : {NUM_WORKERS}")
    print(f"Learning rate      : {LEARNING_RATE}")
    print(f"Loss               : BCE + Dice")
    print(f"AMP                : Enabled")
    print(f"Target epochs      : {target_epochs}")

    if RUN_SANITY_ONLY:
        print()
        print("MODE: 1-EPOCH SANITY RUN")
        print("After verification, set RUN_SANITY_ONLY = False.")

    print("=" * 70)

    # -------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------

    print()
    print("Loading training dataset...")

    _, train_loader = create_dataloader(
        split="train",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    print()
    print("Loading validation dataset...")

    _, val_loader = create_dataloader(
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    print()
    print("=" * 70)
    print("DATA LOADERS READY")
    print("=" * 70)
    print(f"Training batches    : {len(train_loader)}")
    print(f"Validation batches  : {len(val_loader)}")
    print("=" * 70)

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    print()
    print("Creating U-Net...")

    model = UNet(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        base_features=BASE_FEATURES,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)
    print(f"Parameters : {parameter_count:,}")
    print(f"Parameters : {parameter_count / 1e6:.2f} M")
    print("=" * 70)

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------

    criterion = BCEDiceLoss()

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------
    # Scheduler
    # -------------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    # -------------------------------------------------------------
    # AMP scaler
    # -------------------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=True,
    )

    # -------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------

    if RESUME_TRAINING:

        (
            start_epoch,
            best_dice,
            epochs_without_improvement,
            history,
        ) = load_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    else:

        start_epoch = 1
        best_dice = -1.0
        epochs_without_improvement = 0
        history = []

    # -------------------------------------------------------------
    # If checkpoint already finished
    # -------------------------------------------------------------

    if start_epoch > target_epochs:

        print()
        print("=" * 70)
        print("TRAINING ALREADY COMPLETE FOR CURRENT TARGET")
        print("=" * 70)
        print(f"Checkpoint epoch : {start_epoch - 1}")
        print(f"Target epochs    : {target_epochs}")
        print()
        print(
            "If you want to continue toward 50 epochs, "
            "set RUN_SANITY_ONLY = False."
        )
        print("=" * 70)

        return

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        start_epoch,
        target_epochs + 1,
    ):

        print()
        print()
        print("#" * 70)
        print(
            f"EPOCH {epoch}/{target_epochs}"
        )
        print("#" * 70)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        # ---------------------------------------------------------
        # Epoch timer
        # ---------------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_start = time.time()

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------

        print()
        print("Training...")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=target_epochs,
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        print()
        print("Validation...")

        val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ---------------------------------------------------------
        # Scheduler
        # ---------------------------------------------------------

        scheduler.step(val_loss)

        # ---------------------------------------------------------
        # Timing
        # ---------------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_start
        epoch_time_minutes = epoch_time / 60

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        dice = metrics["dice"]
        iou = metrics["iou"]
        precision = metrics["precision"]
        recall = metrics["recall"]

        new_lr = optimizer.param_groups[0]["lr"]

        # ---------------------------------------------------------
        # Print epoch summary
        # ---------------------------------------------------------

        print()
        print("=" * 70)
        print(f"EPOCH {epoch} RESULTS")
        print("=" * 70)

        print(f"Train Loss       : {train_loss:.6f}")
        print(f"Validation Loss  : {val_loss:.6f}")

        print()
        print(f"Dice             : {dice:.6f}")
        print(f"IoU              : {iou:.6f}")
        print(f"Precision        : {precision:.6f}")
        print(f"Recall           : {recall:.6f}")

        print()
        print(f"Learning Rate    : {new_lr:.8f}")
        print(
            f"Epoch Time       : "
            f"{epoch_time_minutes:.2f} minutes"
        )

        # ---------------------------------------------------------
        # History
        # ---------------------------------------------------------

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "learning_rate": new_lr,
            "epoch_time_minutes": epoch_time_minutes,
        }

        history.append(history_row)

        save_history_csv(history)

        # ---------------------------------------------------------
        # Best model
        # ---------------------------------------------------------

        improved = (
            dice > best_dice + MIN_IMPROVEMENT
        )

        if improved:

            best_dice = dice
            epochs_without_improvement = 0

            save_best_model(
                model=model,
                epoch=epoch,
                dice=dice,
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No significant Dice improvement."
            )
            print(
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ---------------------------------------------------------
        # Latest checkpoint
        # ---------------------------------------------------------

        save_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_dice=best_dice,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            history=history,
        )

        print()
        print(
            f"Latest checkpoint saved:"
        )
        print(
            LATEST_CHECKPOINT_PATH
        )

        # ---------------------------------------------------------
        # GPU memory
        # ---------------------------------------------------------

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print()
            print(
                f"GPU memory allocated : "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU memory reserved  : "
                f"{reserved:.2f} GB"
            )

        print("=" * 70)

        # ---------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------

        if (
            not RUN_SANITY_ONLY
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print("=" * 70)
            print("EARLY STOPPING")
            print("=" * 70)
            print(
                f"No improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            print(
                f"Best Dice: {best_dice:.6f}"
            )
            print("=" * 70)

            break

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print()
    print("=" * 70)
    print("TRAINING RUN FINISHED")
    print("=" * 70)

    print(
        f"Best validation Dice : "
        f"{best_dice:.6f}"
    )

    print()
    print("Files:")

    print(
        f"Best model           : "
        f"{BEST_MODEL_PATH}"
    )
 
    print(
        f"Latest checkpoint    : "
        f"{LATEST_CHECKPOINT_PATH}"
    )

    print(
        f"Training history     : "
        f"{HISTORY_PATH}"
    )

    print("=" * 70)

    # -------------------------------------------------------------
    # Sanity mode instructions
    # -------------------------------------------------------------

    if RUN_SANITY_ONLY:

        print()
        print("=" * 70)
        print("SANITY RUN COMPLETE")
        print("=" * 70)

        print(
            "If the metrics and checkpoint files look correct:"
        )

        print()
        print(
            "1. Open train.py"
        )

        print(
            "2. Change:"
        )

        print(
            "       RUN_SANITY_ONLY = True"
        )

        print(
            "   to:"
        )

        print(
            "       RUN_SANITY_ONLY = False"
        )

        print()
        print(
            "3. Run train.py again."
        )

        print()
        print(
            "The script will automatically resume "
            "from latest_checkpoint.pth."
        )

        print("=" * 70)


# =====================================================================
# WINDOWS / SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()