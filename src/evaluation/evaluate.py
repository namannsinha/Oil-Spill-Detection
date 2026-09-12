from pathlib import Path
import csv
import time

import numpy as np
import rasterio
import torch
from torch.cuda.amp import autocast

import sys

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

TEST_ROOT = Path(
    r"D:\SIH_Dataset\Part_III\02_Test_images_and_ground_truth"
)

IMAGE_ROOT = TEST_ROOT / "Images"
MASK_ROOT = TEST_ROOT / "Mask"

MODEL_PATH = PROJECT_ROOT / "models" / "best_unet.pth"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
PROBABILITY_ROOT = OUTPUT_ROOT / "probabilities"
VISUALIZATION_ROOT = OUTPUT_ROOT / "visualizations"

PATCH_SIZE = 512
OVERLAP = 128
STRIDE = PATCH_SIZE - OVERLAP

THRESHOLD = 0.5

BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Exact normalization used during training
VV_MIN = -44.086868
VV_MAX = -12.757253

VH_MIN = -35.326511
VH_MAX = -6.280492


CATEGORIES = [
    "Oil",
    "No oil",
    "Lookalike",
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PROBABILITY_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

VISUALIZATION_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

print("=" * 75)
print("SIH 2026 OIL SPILL DETECTION - PART III EVALUATION")
print("=" * 75)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Model: {MODEL_PATH}")
print(f"Threshold: {THRESHOLD}")
print(f"Patch size: {PATCH_SIZE}")
print(f"Overlap: {OVERLAP}")
print(f"Stride: {STRIDE}")
print("=" * 75)


# ============================================================
# MODEL
# ============================================================

def load_model():
    """
    Load the trained U-Net checkpoint.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model = UNet(
        input_channels=2,
        output_channels=1,
        base_features=64,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle both plain state_dict and checkpoint dictionaries
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("\nModel loaded successfully.")

    return model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sar(image):
    """
    Apply the exact fixed normalization used during training.

    Input:
        image -> [2, H, W]

    Channel 0 = VV
    Channel 1 = VH
    """

    image = image.astype(
        np.float32,
        copy=True
    )

    vv = image[0]
    vh = image[1]

    # Invalid pixels were defined during preprocessing as:
    # VV == 0 OR VH == 0
    invalid = (
        (vv == 0) |
        (vh == 0)
    )

    # VV normalization
    vv = np.clip(
        vv,
        VV_MIN,
        VV_MAX
    )

    vv = (
        (vv - VV_MIN) /
        (VV_MAX - VV_MIN)
    )

    # VH normalization
    vh = np.clip(
        vh,
        VH_MIN,
        VH_MAX
    )

    vh = (
        (vh - VH_MIN) /
        (VH_MAX - VH_MIN)
    )

    result = np.stack(
        [vv, vh],
        axis=0
    ).astype(np.float32)

    # Match preprocessing:
    # invalid pixels become zero
    result[:, invalid] = 0.0

    return result, invalid


# ============================================================
# WINDOW GENERATION
# ============================================================

def get_window_starts(size):
    """
    Generate sliding-window start positions.

    For 2048 with patch=512 and stride=384:

        0
        384
        768
        1152
        1536

    giving exactly 5 positions.
    """

    starts = list(
        range(
            0,
            size - PATCH_SIZE + 1,
            STRIDE
        )
    )

    final_start = size - PATCH_SIZE

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def generate_windows(height, width):

    rows = get_window_starts(height)
    cols = get_window_starts(width)

    windows = []

    for row in rows:

        for col in cols:

            windows.append(
                (row, col)
            )

    return windows


# ============================================================
# SLIDING WINDOW INFERENCE
# ============================================================

@torch.no_grad()
def predict_scene(
    model,
    image
):
    """
    Run sliding-window inference over a complete scene.

    Returns:
        probability_map
        invalid_mask
    """

    normalized, invalid_mask = normalize_sar(
        image
    )

    _, height, width = normalized.shape

    windows = generate_windows(
        height,
        width
    )

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Process patches in batches
    # --------------------------------------------------------

    for batch_start in range(
        0,
        len(windows),
        BATCH_SIZE
    ):

        batch_windows = windows[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch = []

        for row, col in batch_windows:

            patch = normalized[
                :,
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            batch.append(patch)

        batch = np.stack(
            batch,
            axis=0
        )

        tensor = torch.from_numpy(
            batch
        ).float().to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Mixed precision inference
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            with autocast():

                logits = model(
                    tensor
                )

        else:

            logits = model(
                tensor
            )

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = (
            probabilities
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Stitch predictions
        # ----------------------------------------------------

        for (
            (row, col),
            probability
        ) in zip(
            batch_windows,
            probabilities
        ):

            probability_sum[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ] += probability

            prediction_count[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ] += 1.0

    # --------------------------------------------------------
    # Average overlapping predictions
    # --------------------------------------------------------

    probability_map = (
        probability_sum /
        np.maximum(
            prediction_count,
            1.0
        )
    )

    # Invalid pixels must not produce predictions
    probability_map[
        invalid_mask
    ] = 0.0

    return (
        probability_map,
        invalid_mask
    )


# ============================================================
# METRICS
# ============================================================

def calculate_confusion(
    prediction,
    ground_truth,
    valid_mask
):
    """
    Calculate TP, FP, FN, TN.

    Invalid pixels are excluded.
    """

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)
    valid_mask = valid_mask.astype(bool)

    prediction = prediction[valid_mask]
    ground_truth = ground_truth[valid_mask]

    tp = np.sum(
        prediction & ground_truth
    )

    fp = np.sum(
        prediction & ~ground_truth
    )

    fn = np.sum(
        ~prediction & ground_truth
    )

    tn = np.sum(
        ~prediction & ~ground_truth
    )

    return (
        int(tp),
        int(fp),
        int(fn),
        int(tn)
    )


def calculate_metrics(
    tp,
    fp,
    fn,
    tn
):

    dice_denominator = (
        2 * tp +
        fp +
        fn
    )

    iou_denominator = (
        tp +
        fp +
        fn
    )

    precision_denominator = (
        tp + fp
    )

    recall_denominator = (
        tp + fn
    )

    dice = (
        (2 * tp) /
        dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    iou = (
        tp /
        iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    precision = (
        tp /
        precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        tp /
        recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# READ IMAGE + MASK
# ============================================================

def load_test_pair(
    category,
    image_name
):

    image_path = (
        IMAGE_ROOT /
        category /
        image_name
    )

    mask_name = (
        Path(image_name).stem +
        "_segmentation.tif"
    )

    mask_path = (
        MASK_ROOT /
        category /
        mask_name
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Mask not found:\n{mask_path}"
        )

    with rasterio.open(
        image_path
    ) as src:

        image = src.read()

        profile = src.profile.copy()

    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if image.shape[0] != 2:

        raise ValueError(
            f"Expected 2 image bands, "
            f"got {image.shape[0]}:\n"
            f"{image_path}"
        )

    if image.shape[1:] != mask.shape:

        raise ValueError(
            "Image/mask dimensions do not match:\n"
            f"Image: {image.shape}\n"
            f"Mask: {mask.shape}\n"
            f"Scene: {image_name}"
        )

    return (
        image,
        mask,
        profile
    )


# ============================================================
# SAVE GEOTIFF
# ============================================================

def save_prediction(
    path,
    array,
    profile,
    dtype,
    nodata=None
):

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        compress="deflate",
        predictor=2,
    )

    if nodata is not None:
        output_profile.update(
            nodata=nodata
        )

    with rasterio.open(
        path,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            array.astype(dtype),
            1
        )


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    category,
    image_name,
    image,
    ground_truth,
    probability,
    prediction
):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "matplotlib not installed; "
            "skipping visualization."
        )

        return

    vv = image[0]
    vh = image[1]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes[0, 0].imshow(
        vv,
        cmap="gray"
    )

    axes[0, 0].set_title(
        "VV"
    )

    axes[0, 1].imshow(
        vh,
        cmap="gray"
    )

    axes[0, 1].set_title(
        "VH"
    )

    axes[0, 2].imshow(
        ground_truth,
        cmap="gray"
    )

    axes[0, 2].set_title(
        "Ground Truth"
    )

    axes[1, 0].imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[1, 0].set_title(
        "Prediction Probability"
    )

    axes[1, 1].imshow(
        prediction,
        cmap="gray"
    )

    axes[1, 1].set_title(
        f"Prediction @ {THRESHOLD}"
    )

    # Overlay
    axes[1, 2].imshow(
        vv,
        cmap="gray"
    )

    axes[1, 2].imshow(
        ground_truth,
        cmap="Reds",
        alpha=0.35
    )

    axes[1, 2].imshow(
        prediction,
        cmap="Blues",
        alpha=0.35
    )

    axes[1, 2].set_title(
        "Ground Truth / Prediction"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{category} - {image_name}",
        fontsize=14
    )

    fig.tight_layout()

    output_path = (
        VISUALIZATION_ROOT /
        f"{category.replace(' ', '_')}_{Path(image_name).stem}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# EVALUATE ONE SCENE
# ============================================================

def evaluate_scene(
    model,
    category,
    image_name
):

    image, ground_truth, profile = load_test_pair(
        category,
        image_name
    )

    probability, invalid_mask = predict_scene(
        model,
        image
    )

    prediction = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    # Ground truth should be binary
    ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)

    valid_mask = ~invalid_mask

    (
        tp,
        fp,
        fn,
        tn
    ) = calculate_confusion(
        prediction,
        ground_truth,
        valid_mask
    )

    metrics = calculate_metrics(
        tp,
        fp,
        fn,
        tn
    )

    valid_pixels = int(
        np.sum(valid_mask)
    )

    predicted_oil_pixels = int(
        np.sum(
            prediction[
                valid_mask
            ]
        )
    )

    actual_oil_pixels = int(
        np.sum(
            ground_truth[
                valid_mask
            ]
        )
    )

    predicted_oil_percentage = (
        predicted_oil_pixels /
        valid_pixels *
        100
        if valid_pixels > 0
        else 0
    )

    actual_oil_percentage = (
        actual_oil_pixels /
        valid_pixels *
        100
        if valid_pixels > 0
        else 0
    )

    # Save prediction
    prediction_path = (
        PREDICTION_ROOT /
        category.replace(" ", "_")
    )

    prediction_path.mkdir(
        parents=True,
        exist_ok=True
    )

    save_prediction(
        prediction_path /
        image_name,
        prediction,
        profile,
        "uint8",
        nodata=255
    )

    # Save probability
    probability_path = (
        PROBABILITY_ROOT /
        category.replace(" ", "_")
    )

    probability_path.mkdir(
        parents=True,
        exist_ok=True
    )

    save_prediction(
        probability_path /
        image_name,
        probability,
        profile,
        "float32"
    )

    metrics.update(
        {
            "category": category,
            "scene": image_name,
            "valid_pixels": valid_pixels,
            "actual_oil_pixels": actual_oil_pixels,
            "predicted_oil_pixels": predicted_oil_pixels,
            "actual_oil_percentage": actual_oil_percentage,
            "predicted_oil_percentage": predicted_oil_percentage,
            "false_positive_scene": int(
                category != "Oil" and
                predicted_oil_pixels > 0
            ),
        }
    )

    return (
        metrics,
        image,
        ground_truth,
        probability,
        prediction
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows,
    path
):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# AGGREGATE METRICS
# ============================================================

def aggregate_metrics(rows):

    tp = sum(
        row["tp"]
        for row in rows
    )

    fp = sum(
        row["fp"]
        for row in rows
    )

    fn = sum(
        row["fn"]
        for row in rows
    )

    tn = sum(
        row["tn"]
        for row in rows
    )

    metrics = calculate_metrics(
        tp,
        fp,
        fn,
        tn
    )

    metrics["scenes"] = len(rows)

    metrics["mean_scene_dice"] = float(
        np.mean(
            [
                row["dice"]
                for row in rows
            ]
        )
    )

    metrics["mean_scene_iou"] = float(
        np.mean(
            [
                row["iou"]
                for row in rows
            ]
        )
    )

    metrics["mean_scene_precision"] = float(
        np.mean(
            [
                row["precision"]
                for row in rows
            ]
        )
    )

    metrics["mean_scene_recall"] = float(
        np.mean(
            [
                row["recall"]
                for row in rows
            ]
        )
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    model = load_model()

    all_results = []

    category_results = {}

    # --------------------------------------------------------
    # Evaluate each category
    # --------------------------------------------------------

    for category in CATEGORIES:

        print("\n")
        print("=" * 75)
        print(f"EVALUATING: {category}")
        print("=" * 75)

        image_directory = (
            IMAGE_ROOT /
            category
        )

        image_files = sorted(
            image_directory.glob(
                "*.tif"
            )
        )

        if not image_files:

            raise RuntimeError(
                f"No TIFF images found:\n"
                f"{image_directory}"
            )

        print(
            f"Scenes: {len(image_files)}"
        )

        category_rows = []

        category_start = time.time()

        for index, image_path in enumerate(
            image_files,
            start=1
        ):

            image_name = image_path.name

            scene_start = time.time()

            (
                metrics,
                image,
                ground_truth,
                probability,
                prediction
            ) = evaluate_scene(
                model,
                category,
                image_name
            )

            category_rows.append(
                metrics
            )

            all_results.append(
                metrics
            )

            elapsed = (
                time.time() -
                scene_start
            )

            print(
                f"[{index:03d}/{len(image_files):03d}] "
                f"{image_name} | "
                f"Dice={metrics['dice']:.4f} | "
                f"IoU={metrics['iou']:.4f} | "
                f"P={metrics['precision']:.4f} | "
                f"R={metrics['recall']:.4f} | "
                f"time={elapsed:.1f}s"
            )

            # Save representative visualizations
            #
            # 3 scenes per category:
            # first, middle, last
            if index in {
                1,
                len(image_files) // 2,
                len(image_files)
            }:

                save_visualization(
                    category,
                    image_name,
                    image,
                    ground_truth,
                    probability,
                    prediction
                )

        category_results[
            category
        ] = category_rows

        category_time = (
            time.time() -
            category_start
        )

        print(
            f"\n{category} completed "
            f"in {category_time / 60:.2f} minutes."
        )

    # --------------------------------------------------------
    # Save per-scene CSV
    # --------------------------------------------------------

    scene_csv = (
        OUTPUT_ROOT /
        "scene_metrics.csv"
    )

    save_csv(
        all_results,
        scene_csv
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall = aggregate_metrics(
        all_results
    )

    # --------------------------------------------------------
    # Category metrics
    # --------------------------------------------------------

    summary_rows = []

    for category in CATEGORIES:

        rows = category_results[
            category
        ]

        aggregate = aggregate_metrics(
            rows
        )

        false_positive_scenes = sum(
            row["false_positive_scene"]
            for row in rows
        )

        mean_predicted_oil = np.mean(
            [
                row[
                    "predicted_oil_percentage"
                ]
                for row in rows
            ]
        )

        aggregate.update(
            {
                "category": category,
                "false_positive_scenes":
                    false_positive_scenes,
                "false_positive_scene_rate":
                    (
                        false_positive_scenes /
                        len(rows) *
                        100
                    ),
                "mean_predicted_oil_percentage":
                    float(
                        mean_predicted_oil
                    ),
            }
        )

        summary_rows.append(
            aggregate
        )

    # Overall row
    overall.update(
        {
            "category": "Overall",
            "false_positive_scenes": "",
            "false_positive_scene_rate": "",
            "mean_predicted_oil_percentage":
                float(
                    np.mean(
                        [
                            row[
                                "predicted_oil_percentage"
                            ]
                            for row in all_results
                        ]
                    )
                ),
        }
    )

    summary_rows.append(
        overall
    )

    summary_csv = (
        OUTPUT_ROOT /
        "test_metrics.csv"
    )

    save_csv(
        summary_rows,
        summary_csv
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    total_time = (
        time.time() -
        start_time
    )

    print("\n")
    print("=" * 75)
    print("FINAL TEST RESULTS")
    print("=" * 75)

    for row in summary_rows:

        print(
            f"\n{row['category']}"
        )

        print(
            f"  Scenes:     {row['scenes']}"
        )

        print(
            f"  Dice:       {row['dice']:.4f}"
        )

        print(
            f"  IoU:        {row['iou']:.4f}"
        )

        print(
            f"  Precision:  {row['precision']:.4f}"
        )

        print(
            f"  Recall:     {row['recall']:.4f}"
        )

        print(
            f"  TP:         {row['tp']:,}"
        )

        print(
            f"  FP:         {row['fp']:,}"
        )

        print(
            f"  FN:         {row['fn']:,}"
        )

        if row["category"] in ["No oil", "Lookalike"]:

            print(
                f"  FP scenes:  "
                f"{row['false_positive_scenes']}"
                f"/{row['scenes']} "
                f"({row['false_positive_scene_rate']:.2f}%)"
            )

            print(
                f"  Mean predicted oil: "
                f"{row['mean_predicted_oil_percentage']:.4f}%"
            )

    print("\n")
    print("=" * 75)
    print(
        f"Total evaluation time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"\nScene metrics:"
        f"\n{scene_csv}"
    )

    print(
        f"\nSummary metrics:"
        f"\n{summary_csv}"
    )

    print(
        f"\nPredictions:"
        f"\n{PREDICTION_ROOT}"
    )

    print(
        f"\nProbabilities:"
        f"\n{PROBABILITY_ROOT}"
    )

    print(
        f"\nVisualizations:"
        f"\n{VISUALIZATION_ROOT}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()