"""
Threshold Analysis for Oil Spill Detection

Uses already-saved probability maps from evaluate.py.
No model inference or GPU is required.

Tests multiple thresholds and compares:
- Oil Dice
- Oil IoU
- Oil Precision
- Oil Recall
- No-Oil False Positive Scene Rate
- Lookalike False Positive Scene Rate
- Mean predicted oil percentage
"""

from pathlib import Path
import csv
import numpy as np
import rasterio

PROBABILITY_CATEGORIES = {
    "Oil": "Oil",
    "No oil": "No_oil",
    "Lookalike": "Lookalike",
}

MASK_CATEGORIES = {
    "Oil": "Oil",
    "No oil": "No oil",
    "Lookalike": "Lookalike",
}

IMAGE_CATEGORIES = {
    "Oil": "Oil",
    "No oil": "No oil",
    "Lookalike": "Lookalike",
}

def find_probability_file(category, scene_name):
    category_dir = PROBABILITY_ROOT / PROBABILITY_CATEGORIES[category]

    probability_path = category_dir / f"{scene_name}.tif"

    if probability_path.exists():
        return probability_path

    return None

def find_mask_file(category, scene_name):
    category_dir = TEST_ROOT / "Mask" / MASK_CATEGORIES[category]

    mask_path = category_dir / f"{scene_name}.tif"

    if mask_path.exists():
        return mask_path

    return None


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = Path(
    r"D:\SIH_Dataset\Part_III\02_Test_images_and_ground_truth"
)

EVALUATION_ROOT = PROJECT_ROOT / "outputs" / "evaluation"

PROBABILITY_ROOT = EVALUATION_ROOT / "probabilities"

OUTPUT_DIR = EVALUATION_ROOT / "threshold_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "threshold_results.csv"

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# CATEGORY PATHS
# ============================================================

CATEGORIES = {
    "Oil": "Oil",
    "No oil": "No oil",
    "Lookalike": "Lookalike",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_probability_file(category, scene_name):
    """
    Find the saved probability GeoTIFF for a scene.
    """

    category_dir = PROBABILITY_ROOT / CATEGORIES[category]

    probability_path = category_dir / f"{scene_name}.tif"

    if probability_path.exists():
        return probability_path

    return None


def load_raster(path):
    """Load the first raster band as float32."""

    with rasterio.open(path) as src:
        data = src.read(1)

    return data.astype(np.float32)


def calculate_confusion(prediction, target):
    """
    Calculate TP, FP, FN, TN.
    """

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()

    return int(tp), int(fp), int(fn), int(tn)


def calculate_metrics(tp, fp, fn):
    """
    Calculate Dice, IoU, Precision and Recall.

    For an empty prediction and empty target:
    Dice and IoU are treated as 1.
    """

    denominator_dice = 2 * tp + fp + fn

    if denominator_dice == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / denominator_dice

    denominator_iou = tp + fp + fn

    if denominator_iou == 0:
        iou = 1.0
    else:
        iou = tp / denominator_iou

    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 1.0
    else:
        precision = tp / precision_denominator

    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 1.0
    else:
        recall = tp / recall_denominator

    return dice, iou, precision, recall


def get_scene_names(category):
    category_dir = TEST_ROOT / "Mask" / MASK_CATEGORIES[category]

    if not category_dir.exists():
        return []

    return sorted(
        p.stem
        for p in category_dir.glob("*.tif")
    )


# ============================================================
# CATEGORY EVALUATION
# ============================================================

def evaluate_category(category, threshold):
    """
    Evaluate one category at one threshold.
    """

    scene_names = get_scene_names(category)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    false_positive_scenes = 0
    total_scenes = 0

    predicted_oil_percentages = []

    missing_probability_files = []
    missing_mask_files = []

    for scene_name in scene_names:

        probability_path = find_probability_file(
            category,
            scene_name
        )

        if probability_path is None:
            missing_probability_files.append(scene_name)
            continue

        mask_path = find_mask_file(
            category,
            scene_name
        )

        if mask_path is None:
            missing_mask_files.append(scene_name)
            continue

        probability = load_raster(probability_path)
        target = load_raster(mask_path)

        # Convert ground truth to binary
        target = target > 0

        # Threshold probability map
        prediction = probability >= threshold

        tp, fp, fn, _ = calculate_confusion(
            prediction,
            target
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_scenes += 1

        predicted_oil_pixels = prediction.sum()
        total_pixels = prediction.size

        predicted_oil_percentage = (
            predicted_oil_pixels / total_pixels
        ) * 100

        predicted_oil_percentages.append(
            predicted_oil_percentage
        )

        # For negative categories, any prediction is a
        # false-positive scene.
        if category in ["No oil", "Lookalike"]:
            if predicted_oil_pixels > 0:
                false_positive_scenes += 1

    dice, iou, precision, recall = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    if predicted_oil_percentages:
        mean_predicted_oil = np.mean(
            predicted_oil_percentages
        )
    else:
        mean_predicted_oil = 0.0

    if category in ["No oil", "Lookalike"] and total_scenes > 0:
        false_positive_scene_rate = (
            false_positive_scenes / total_scenes
        ) * 100
    else:
        false_positive_scene_rate = np.nan

    return {
        "category": category,
        "threshold": threshold,
        "scenes": total_scenes,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "false_positive_scenes": false_positive_scenes,
        "false_positive_scene_rate": false_positive_scene_rate,
        "mean_predicted_oil_percentage": mean_predicted_oil,
        "missing_probability_files": len(
            missing_probability_files
        ),
        "missing_mask_files": len(
            missing_mask_files
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OIL SPILL DETECTION - THRESHOLD ANALYSIS")
    print("=" * 70)

    print(f"Probability maps : {PROBABILITY_ROOT}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Thresholds       : {THRESHOLDS}")
    print()

    all_results = []

    for threshold in THRESHOLDS:

        print("-" * 70)
        print(f"THRESHOLD = {threshold:.2f}")
        print("-" * 70)

        for category in ["Oil", "No oil", "Lookalike"]:

            result = evaluate_category(
                category,
                threshold
            )

            all_results.append(result)

            print(
                f"{category:10s} | "
                f"Dice={result['dice']:.4f} | "
                f"IoU={result['iou']:.4f} | "
                f"Precision={result['precision']:.4f} | "
                f"Recall={result['recall']:.4f}"
            )

            if category in ["No oil", "Lookalike"]:

                print(
                    f"{'':10s} | "
                    f"FP scenes="
                    f"{result['false_positive_scenes']}/"
                    f"{result['scenes']} "
                    f"({result['false_positive_scene_rate']:.2f}%) | "
                    f"Mean predicted oil="
                    f"{result['mean_predicted_oil_percentage']:.4f}%"
                )

        print()

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "category",
        "threshold",
        "scenes",
        "dice",
        "iou",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "false_positive_scenes",
        "false_positive_scene_rate",
        "mean_predicted_oil_percentage",
        "missing_probability_files",
        "missing_mask_files",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    print("=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"Results saved to:")
    print(OUTPUT_CSV)

    # ========================================================
    # FIND BEST OIL THRESHOLD
    # ========================================================

    oil_results = [
        r for r in all_results
        if r["category"] == "Oil"
    ]

    best_oil = max(
        oil_results,
        key=lambda x: x["dice"]
    )

    print()
    print("BEST THRESHOLD BY OIL DICE")
    print("-" * 70)

    print(
        f"Threshold : {best_oil['threshold']:.2f}"
    )
    print(
        f"Dice      : {best_oil['dice']:.4f}"
    )
    print(
        f"IoU       : {best_oil['iou']:.4f}"
    )
    print(
        f"Precision : {best_oil['precision']:.4f}"
    )
    print(
        f"Recall    : {best_oil['recall']:.4f}"
    )

    # ========================================================
    # LOOKALIKE COMPARISON
    # ========================================================

    lookalike_results = [
        r for r in all_results
        if r["category"] == "Lookalike"
    ]

    print()
    print("LOOKALIKE FALSE-POSITIVE ANALYSIS")
    print("-" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'FP Scene %':<15}"
        f"{'Mean Pred Oil %':<18}"
    )

    for result in lookalike_results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['false_positive_scene_rate']:<15.2f}"
            f"{result['mean_predicted_oil_percentage']:<18.4f}"
        )

    # ========================================================
    # NO-OIL COMPARISON
    # ========================================================

    no_oil_results = [
        r for r in all_results
        if r["category"] == "No oil"
    ]

    print()
    print("NO-OIL FALSE-POSITIVE ANALYSIS")
    print("-" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'FP Scene %':<15}"
        f"{'Mean Pred Oil %':<18}"
    )

    for result in no_oil_results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['false_positive_scene_rate']:<15.2f}"
            f"{result['mean_predicted_oil_percentage']:<18.4f}"
        )


if __name__ == "__main__":
    main()