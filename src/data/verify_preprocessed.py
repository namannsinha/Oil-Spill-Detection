from pathlib import Path
import csv
import random

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "patch_manifest.csv"
)

VISUALIZATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_PATCH_SIZE = 512
EXPECTED_IMAGE_CHANNELS = 2

DATASETS = [
    "oil",
    "no_oil",
    "lookalike"
]

SPLITS = [
    "train",
    "validation"
]


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    return rows


# ============================================================
# BASIC HELPERS
# ============================================================

def get_unique_values(
    rows,
    column
):

    return sorted(
        set(
            row[column]
            for row in rows
        )
    )


def count_rows(
    rows,
    dataset=None,
    split=None,
    patch_type=None
):

    count = 0

    for row in rows:

        if (
            dataset is not None
            and row["dataset"] != dataset
        ):
            continue

        if (
            split is not None
            and row["split"] != split
        ):
            continue

        if (
            patch_type is not None
            and row["patch_type"] != patch_type
        ):
            continue

        count += 1

    return count


# ============================================================
# VERIFY MANIFEST
# ============================================================

def verify_manifest(rows):

    print("\n" + "=" * 75)
    print("1. MANIFEST VERIFICATION")
    print("=" * 75)

    print(
        f"Manifest rows: {len(rows):,}"
    )

    required_columns = [
        "dataset",
        "split",
        "source_image",
        "source_mask",
        "patch_name",
        "row",
        "col",
        "patch_size",
        "oil_pixels",
        "valid_pixels",
        "oil_percentage",
        "patch_type",
        "image_path",
        "mask_path",
        "invalid_path"
    ]

    actual_columns = set(
        rows[0].keys()
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in actual_columns
    ]

    if missing_columns:

        print(
            "❌ Missing columns:"
        )

        for column in missing_columns:
            print(
                f"   {column}"
            )

        return False

    print(
        "✅ All required manifest columns present"
    )

    datasets = get_unique_values(
        rows,
        "dataset"
    )

    splits = get_unique_values(
        rows,
        "split"
    )

    patch_types = get_unique_values(
        rows,
        "patch_type"
    )

    print(
        f"Datasets: {datasets}"
    )

    print(
        f"Splits: {splits}"
    )

    print(
        f"Patch types: {patch_types}"
    )

    return True


# ============================================================
# VERIFY FILES
# ============================================================

def verify_files(rows):

    print("\n" + "=" * 75)
    print("2. FILE EXISTENCE")
    print("=" * 75)

    missing_images = []
    missing_masks = []
    missing_invalid = []

    for index, row in enumerate(rows):

        image_path = Path(
            row["image_path"]
        )

        mask_path = Path(
            row["mask_path"]
        )

        invalid_path = Path(
            row["invalid_path"]
        )

        if not image_path.exists():

            missing_images.append(
                image_path
            )

        if not mask_path.exists():

            missing_masks.append(
                mask_path
            )

        if not invalid_path.exists():

            missing_invalid.append(
                invalid_path
            )

        if (
            (index + 1) % 5000 == 0
        ):

            print(
                f"Checked "
                f"{index + 1:,}/"
                f"{len(rows):,}"
            )

    print(
        f"\nMissing image files: "
        f"{len(missing_images)}"
    )

    print(
        f"Missing mask files: "
        f"{len(missing_masks)}"
    )

    print(
        f"Missing invalid files: "
        f"{len(missing_invalid)}"
    )

    passed = (
        len(missing_images) == 0
        and
        len(missing_masks) == 0
        and
        len(missing_invalid) == 0
    )

    if passed:

        print(
            "✅ All referenced files exist"
        )

    else:

        print(
            "❌ Missing files detected"
        )

    return passed


# ============================================================
# VERIFY PATCH ARRAYS
# ============================================================

def verify_arrays(rows):

    print("\n" + "=" * 75)
    print("3. PATCH ARRAY VALIDATION")
    print("=" * 75)

    errors = []

    image_min = np.inf
    image_max = -np.inf

    total_invalid_pixels = 0
    total_pixels = 0

    total_oil_pixels = 0

    for index, row in enumerate(rows):

        image_path = Path(
            row["image_path"]
        )

        mask_path = Path(
            row["mask_path"]
        )

        invalid_path = Path(
            row["invalid_path"]
        )

        try:

            image = np.load(
                image_path,
                mmap_mode="r",
                allow_pickle=False
            )

            mask = np.load(
                mask_path,
                mmap_mode="r",
                allow_pickle=False
            )

            invalid = np.load(
                invalid_path,
                mmap_mode="r",
                allow_pickle=False
            )

            # ------------------------------------------------
            # Shape checks
            # ------------------------------------------------

            if image.shape != (
                2,
                EXPECTED_PATCH_SIZE,
                EXPECTED_PATCH_SIZE
            ):

                errors.append(
                    (
                        row["patch_name"],
                        "image_shape",
                        image.shape
                    )
                )

            if mask.shape != (
                EXPECTED_PATCH_SIZE,
                EXPECTED_PATCH_SIZE
            ):

                errors.append(
                    (
                        row["patch_name"],
                        "mask_shape",
                        mask.shape
                    )
                )

            if invalid.shape != (
                EXPECTED_PATCH_SIZE,
                EXPECTED_PATCH_SIZE
            ):

                errors.append(
                    (
                        row["patch_name"],
                        "invalid_shape",
                        invalid.shape
                    )
                )

            # ------------------------------------------------
            # Dtype checks
            # ------------------------------------------------

            if image.dtype != np.float32:

                errors.append(
                    (
                        row["patch_name"],
                        "image_dtype",
                        str(image.dtype)
                    )
                )

            if mask.dtype != np.uint8:

                errors.append(
                    (
                        row["patch_name"],
                        "mask_dtype",
                        str(mask.dtype)
                    )
                )

            if invalid.dtype != np.uint8:

                errors.append(
                    (
                        row["patch_name"],
                        "invalid_dtype",
                        str(invalid.dtype)
                    )
                )

            # ------------------------------------------------
            # Image range
            # ------------------------------------------------

            current_min = float(
                np.min(image)
            )

            current_max = float(
                np.max(image)
            )

            image_min = min(
                image_min,
                current_min
            )

            image_max = max(
                image_max,
                current_max
            )

            if current_min < 0:

                errors.append(
                    (
                        row["patch_name"],
                        "image_below_zero",
                        current_min
                    )
                )

            if current_max > 1:

                errors.append(
                    (
                        row["patch_name"],
                        "image_above_one",
                        current_max
                    )
                )

            # ------------------------------------------------
            # Mask values
            # ------------------------------------------------

            mask_values = np.unique(
                mask
            )

            if not np.all(
                np.isin(
                    mask_values,
                    [0, 1]
                )
            ):

                errors.append(
                    (
                        row["patch_name"],
                        "invalid_mask_values",
                        mask_values.tolist()
                    )
                )

            # ------------------------------------------------
            # Invalid mask values
            # ------------------------------------------------

            invalid_values = np.unique(
                invalid
            )

            if not np.all(
                np.isin(
                    invalid_values,
                    [0, 1]
                )
            ):

                errors.append(
                    (
                        row["patch_name"],
                        "invalid_mask_values",
                        invalid_values.tolist()
                    )
                )

            # ------------------------------------------------
            # Image/mask consistency
            # ------------------------------------------------

            manifest_oil_pixels = int(
                float(
                    row["oil_pixels"]
                )
            )

            actual_oil_pixels = int(
                np.sum(
                    mask == 1
                )
            )

            if (
                manifest_oil_pixels
                != actual_oil_pixels
            ):

                errors.append(
                    (
                        row["patch_name"],
                        "oil_pixel_mismatch",
                        (
                            manifest_oil_pixels,
                            actual_oil_pixels
                        )
                    )
                )

            # ------------------------------------------------
            # Invalid statistics
            # ------------------------------------------------

            invalid_pixels = int(
                np.sum(
                    invalid == 1
                )
            )

            total_invalid_pixels += (
                invalid_pixels
            )

            total_pixels += (
                EXPECTED_PATCH_SIZE
                * EXPECTED_PATCH_SIZE
            )

            total_oil_pixels += (
                actual_oil_pixels
            )

        except Exception as exc:

            errors.append(
                (
                    row["patch_name"],
                    "load_error",
                    str(exc)
                )
            )

        if (
            (index + 1) % 5000 == 0
            or
            (index + 1) == len(rows)
        ):

            print(
                f"Validated arrays: "
                f"{index + 1:,}/"
                f"{len(rows):,}"
            )

    print(
        f"\nGlobal normalized minimum: "
        f"{image_min:.6f}"
    )

    print(
        f"Global normalized maximum: "
        f"{image_max:.6f}"
    )

    invalid_percentage = (
        total_invalid_pixels
        / total_pixels
        * 100
    )

    oil_percentage = (
        total_oil_pixels
        / total_pixels
        * 100
    )

    print(
        f"Overall invalid pixels: "
        f"{invalid_percentage:.4f}%"
    )

    print(
        f"Overall oil pixels: "
        f"{oil_percentage:.4f}%"
    )

    print(
        f"\nArray errors: "
        f"{len(errors)}"
    )

    if errors:

        print(
            "\n❌ Array validation failed."
        )

        print(
            "\nFirst 10 errors:"
        )

        for error in errors[:10]:

            print(
                f"  {error}"
            )

        return False

    print(
        "✅ All patch arrays passed validation"
    )

    return True


# ============================================================
# VERIFY TRAIN / VALIDATION LEAKAGE
# ============================================================

def verify_scene_leakage(rows):

    print("\n" + "=" * 75)
    print("4. TRAIN / VALIDATION LEAKAGE")
    print("=" * 75)

    train_scenes = set(
        row["source_image"]
        for row in rows
        if row["split"] == "train"
    )

    validation_scenes = set(
        row["source_image"]
        for row in rows
        if row["split"] == "validation"
    )

    overlap = (
        train_scenes
        &
        validation_scenes
    )

    print(
        f"Train scenes: "
        f"{len(train_scenes):,}"
    )

    print(
        f"Validation scenes: "
        f"{len(validation_scenes):,}"
    )

    print(
        f"Overlapping scenes: "
        f"{len(overlap):,}"
    )

    if overlap:

        print(
            "\n❌ DATA LEAKAGE DETECTED"
        )

        for scene in list(
            overlap
        )[:10]:

            print(
                f"  {scene}"
            )

        return False

    print(
        "✅ No train/validation scene leakage"
    )

    return True


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

def print_distribution(rows):

    print("\n" + "=" * 75)
    print("5. DATASET DISTRIBUTION")
    print("=" * 75)

    for dataset in DATASETS:

        print(
            f"\n{dataset.upper()}"
        )

        dataset_rows = [
            row
            for row in rows
            if row["dataset"] == dataset
        ]

        scenes = set(
            row["source_image"]
            for row in dataset_rows
        )

        print(
            f"Scenes: "
            f"{len(scenes):,}"
        )

        print(
            f"Total patches: "
            f"{len(dataset_rows):,}"
        )

        for split in SPLITS:

            split_rows = [
                row
                for row in dataset_rows
                if row["split"] == split
            ]

            print(
                f"  {split}: "
                f"{len(split_rows):,}"
            )

        oil = sum(
            row["patch_type"] == "oil"
            for row in dataset_rows
        )

        background = sum(
            row["patch_type"] == "background"
            for row in dataset_rows
        )

        print(
            f"Oil patches: "
            f"{oil:,}"
        )

        print(
            f"Background patches: "
            f"{background:,}"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    oil = sum(
        row["patch_type"] == "oil"
        for row in rows
    )

    background = sum(
        row["patch_type"] == "background"
        for row in rows
    )

    print(
        "\nOVERALL"
    )

    print(
        f"Oil patches: "
        f"{oil:,}"
    )

    print(
        f"Background patches: "
        f"{background:,}"
    )

    print(
        f"Oil-containing patch ratio: "
        f"{oil / len(rows) * 100:.2f}%"
    )

    print(
        f"Background patch ratio: "
        f"{background / len(rows) * 100:.2f}%"
    )


# ============================================================
# SELECT REPRESENTATIVE PATCHES
# ============================================================

def select_representative_patches(rows):

    train_rows = [
        row
        for row in rows
        if row["split"] == "train"
    ]

    oil_rows = [
        row
        for row in train_rows
        if (
            row["dataset"] == "oil"
            and row["patch_type"] == "oil"
        )
    ]

    oil_rows.sort(
        key=lambda row: float(
            row["oil_percentage"]
        )
    )

    examples = []

    # --------------------------------------------------------
    # Small oil
    # --------------------------------------------------------

    if oil_rows:

        examples.append(
            (
                "Small oil",
                oil_rows[
                    len(oil_rows) // 10
                ]
            )
        )

    # --------------------------------------------------------
    # Medium oil
    # --------------------------------------------------------

    if oil_rows:

        examples.append(
            (
                "Medium oil",
                oil_rows[
                    len(oil_rows) // 2
                ]
            )
        )

    # --------------------------------------------------------
    # Large oil
    # --------------------------------------------------------

    if oil_rows:

        examples.append(
            (
                "Large oil",
                oil_rows[
                    -max(
                        1,
                        len(oil_rows) // 20
                    )
                ]
            )
        )

    # --------------------------------------------------------
    # Oil-scene background
    # --------------------------------------------------------

    oil_background = [
        row
        for row in train_rows
        if (
            row["dataset"] == "oil"
            and row["patch_type"] == "background"
        )
    ]

    if oil_background:

        examples.append(
            (
                "Oil-scene background",
                random.choice(
                    oil_background
                )
            )
        )

    # --------------------------------------------------------
    # No oil
    # --------------------------------------------------------

    no_oil = [
        row
        for row in train_rows
        if row["dataset"] == "no_oil"
    ]

    if no_oil:

        examples.append(
            (
                "No oil",
                random.choice(
                    no_oil
                )
            )
        )

    # --------------------------------------------------------
    # Lookalike
    # --------------------------------------------------------

    lookalike = [
        row
        for row in train_rows
        if row["dataset"] == "lookalike"
    ]

    if lookalike:

        examples.append(
            (
                "Lookalike",
                random.choice(
                    lookalike
                )
            )
        )

    return examples


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_examples(
    rows
):

    print("\n" + "=" * 75)
    print("6. REPRESENTATIVE PATCH VISUALIZATION")
    print("=" * 75)

    examples = (
        select_representative_patches(
            rows
        )
    )

    if not examples:

        print(
            "❌ No examples available."
        )

        return

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    figure, axes = plt.subplots(
        len(examples),
        4,
        figsize=(16, 4 * len(examples))
    )

    if len(examples) == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row_index, (
        title,
        row
    ) in enumerate(examples):

        image = np.load(
            Path(
                row["image_path"]
            ),
            allow_pickle=False
        )

        mask = np.load(
            Path(
                row["mask_path"]
            ),
            allow_pickle=False
        )

        invalid = np.load(
            Path(
                row["invalid_path"]
            ),
            allow_pickle=False
        )

        vv = image[0]
        vh = image[1]

        # ----------------------------------------------------
        # VV
        # ----------------------------------------------------

        axes[
            row_index,
            0
        ].imshow(
            vv,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[
            row_index,
            0
        ].set_title(
            f"{title} - VV"
        )

        # ----------------------------------------------------
        # VH
        # ----------------------------------------------------

        axes[
            row_index,
            1
        ].imshow(
            vh,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[
            row_index,
            1
        ].set_title(
            f"{title} - VH"
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        axes[
            row_index,
            2
        ].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[
            row_index,
            2
        ].set_title(
            (
                f"{title} - Ground Truth\n"
                f"Oil: "
                f"{row['oil_percentage']}%"
            )
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        axes[
            row_index,
            3
        ].imshow(
            vv,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        # Semi-transparent mask overlay
        overlay = np.ma.masked_where(
            mask == 0,
            mask
        )

        axes[
            row_index,
            3
        ].imshow(
            overlay,
            cmap="autumn",
            alpha=0.5,
            vmin=0,
            vmax=1
        )

        axes[
            row_index,
            3
        ].set_title(
            f"{title} - Overlay"
        )

        for column in range(4):

            axes[
                row_index,
                column
            ].axis("off")

    figure.suptitle(
        "Preprocessed Dataset Verification",
        fontsize=16
    )

    figure.tight_layout()

    output_path = (
        VISUALIZATION_DIR
        / "preprocessed_verification.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"✅ Visualization saved:\n"
        f"{output_path}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

def main():

    print("=" * 75)
    print(
        "       PREPROCESSED DATASET VERIFICATION"
    )
    print("=" * 75)

    random.seed(42)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rows = load_manifest()

    print(
        f"\nLoaded "
        f"{len(rows):,} manifest rows."
    )

    # --------------------------------------------------------
    # Run checks
    # --------------------------------------------------------

    manifest_ok = (
        verify_manifest(rows)
    )

    files_ok = (
        verify_files(rows)
    )

    arrays_ok = (
        verify_arrays(rows)
    )

    leakage_ok = (
        verify_scene_leakage(rows)
    )

    print_distribution(
        rows
    )

    visualize_examples(
        rows
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL VERIFICATION RESULT")
    print("=" * 75)

    checks = {
        "Manifest": manifest_ok,
        "Files": files_ok,
        "Arrays": arrays_ok,
        "Scene leakage": leakage_ok
    }

    for name, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        symbol = (
            "✅"
            if passed
            else "❌"
        )

        print(
            f"{symbol} {name}: "
            f"{status}"
        )

    if all(checks.values()):

        print(
            "\n🎉 PREPROCESSED DATASET "
            "PASSED ALL AUTOMATED CHECKS."
        )

        print(
            "\nNext step:"
        )

        print(
            "Build PyTorch Dataset + DataLoader."
        )

    else:

        print(
            "\n⚠️ DATASET VERIFICATION "
            "FOUND ISSUES."
        )

        print(
            "Do not start training yet."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()