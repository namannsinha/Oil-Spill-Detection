from pathlib import Path
import csv
import random

import yaml
import rasterio
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "config.yaml"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PATCH_DIR = (
    PROJECT_ROOT
    / "data"
    / "patches"
)

MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "manifests"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# CONFIG
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r"
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# FIND TIFFS
# ============================================================

def find_tiffs(folder):

    folder = Path(folder)

    return sorted(
        list(folder.rglob("*.tif"))
        +
        list(folder.rglob("*.tiff"))
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    PATCH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_band(
    band,
    minimum,
    maximum,
    invalid_value=0
):
    """
    Normalize one SAR band to [0, 1].

    Exact zero values are treated as invalid.
    """

    band = band.astype(
        np.float32
    )

    invalid_mask = (
        band == invalid_value
    )

    band = np.clip(
        band,
        minimum,
        maximum
    )

    band = (
        band - minimum
    ) / (
        maximum - minimum
    )

    # Protect against floating-point rounding.
    band = np.clip(
        band,
        0.0,
        1.0
    )

    # Invalid pixels are explicitly set to 0.
    band[invalid_mask] = 0.0

    return (
        band,
        invalid_mask
    )


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(
    image,
    config
):
    """
    Normalize VV and VH independently.
    """

    preprocessing = config[
        "preprocessing"
    ]

    normalization = preprocessing[
        "normalization"
    ]

    invalid_value = preprocessing.get(
        "invalid_value",
        0
    )

    vv, vv_invalid = normalize_band(
        image[0],
        normalization["vv_min"],
        normalization["vv_max"],
        invalid_value
    )

    vh, vh_invalid = normalize_band(
        image[1],
        normalization["vh_min"],
        normalization["vh_max"],
        invalid_value
    )

    # A pixel is considered invalid if either
    # polarization contains an invalid value.
    invalid_mask = (
        vv_invalid
        |
        vh_invalid
    )

    normalized = np.stack(
        [
            vv,
            vh
        ],
        axis=0
    )

    return (
        normalized,
        invalid_mask
    )


# ============================================================
# PATCH POSITIONS
# ============================================================

def generate_patch_positions(
    height,
    width,
    patch_size,
    overlap
):
    """
    Generate patch coordinates.

    Returns:
        (row_start, col_start)
    """

    stride = (
        patch_size
        - overlap
    )

    if stride <= 0:

        raise ValueError(
            "Overlap must be smaller "
            "than patch size."
        )

    row_positions = list(
        range(
            0,
            height - patch_size + 1,
            stride
        )
    )

    col_positions = list(
        range(
            0,
            width - patch_size + 1,
            stride
        )
    )

    # --------------------------------------------------------
    # Ensure last patch reaches image boundary
    # --------------------------------------------------------

    last_row = (
        height - patch_size
    )

    last_col = (
        width - patch_size
    )

    if last_row not in row_positions:

        row_positions.append(
            last_row
        )

    if last_col not in col_positions:

        col_positions.append(
            last_col
        )

    row_positions = sorted(
        set(row_positions)
    )

    col_positions = sorted(
        set(col_positions)
    )

    positions = []

    for row in row_positions:

        for col in col_positions:

            positions.append(
                (
                    row,
                    col
                )
            )

    return positions


# ============================================================
# PATCH TYPE
# ============================================================

def classify_patch(
    mask_patch,
    invalid_patch
):
    """
    Classify patch based on its mask.

    oil:
        contains at least one oil pixel

    background:
        no oil pixels

    invalid:
        completely invalid
    """

    oil_pixels = int(
        np.sum(
            mask_patch == 1
        )
    )

    valid_pixels = int(
        np.sum(
            ~invalid_patch
        )
    )

    if valid_pixels == 0:

        return (
            "invalid",
            oil_pixels,
            valid_pixels
        )

    if oil_pixels > 0:

        return (
            "oil",
            oil_pixels,
            valid_pixels
        )

    return (
        "background",
        oil_pixels,
        valid_pixels
    )


# ============================================================
# SAVE PATCH
# ============================================================

def save_patch(
    image_patch,
    mask_patch,
    invalid_patch,
    output_dir,
    patch_name
):
    """
    Save image, mask and invalid-pixel mask.
    """

    output_dir = Path(
        output_dir
    )

    image_dir = (
        output_dir
        / "images"
    )

    mask_dir = (
        output_dir
        / "masks"
    )

    invalid_dir = (
        output_dir
        / "invalid"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    invalid_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        image_dir
        / f"{patch_name}.npy"
    )

    mask_path = (
        mask_dir
        / f"{patch_name}.npy"
    )

    invalid_path = (
        invalid_dir
        / f"{patch_name}.npy"
    )

    np.save(
        image_path,
        image_patch.astype(
            np.float32
        )
    )

    np.save(
        mask_path,
        mask_patch.astype(
            np.uint8
        )
    )

    np.save(
        invalid_path,
        invalid_patch.astype(
            np.uint8
        )
    )

    return (
        image_path,
        mask_path,
        invalid_path
    )


# ============================================================
# PROCESS ONE SCENE
# ============================================================

def process_scene(
    image_path,
    mask_path,
    dataset_name,
    split,
    config,
    manifest_rows
):
    """
    Process one 2048x2048 scene.
    """

    preprocessing = config[
        "preprocessing"
    ]

    patch_size = preprocessing[
        "patch_size"
    ]

    overlap = preprocessing[
        "overlap"
    ]

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    with rasterio.open(
        image_path
    ) as src:

        image = src.read()

    # --------------------------------------------------------
    # Load mask
    # --------------------------------------------------------

    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if image.shape[1:] != mask.shape:

        raise ValueError(
            f"Image/mask shape mismatch: "
            f"{image.shape} vs {mask.shape}"
        )

    height = image.shape[1]
    width = image.shape[2]

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized_image, invalid_mask = (
        normalize_image(
            image,
            config
        )
    )

    # --------------------------------------------------------
    # Generate patches
    # --------------------------------------------------------

    positions = generate_patch_positions(
        height,
        width,
        patch_size,
        overlap
    )

    # --------------------------------------------------------
    # Patch sampling configuration
    # --------------------------------------------------------

    sampling = preprocessing.get(
            "patch_sampling",
            {}
        )

    min_oil_pixels = sampling.get(
        "min_oil_pixels",
        1
    )

    max_background = sampling.get(
        "max_background_patches_per_image",
        5
    )

    max_no_oil = sampling.get(
        "max_no_oil_patches_per_image",
        5
    )

    max_lookalike = sampling.get(
        "max_lookalike_patches_per_image",
        5
    )

    keep_all_oil = sampling.get(
        "keep_all_oil_patches",
        True
    )

    # --------------------------------------------------------
    # Store background candidates
    # --------------------------------------------------------

    background_candidates = []

    oil_candidates = []

    patch_records = []

    # --------------------------------------------------------
    # Analyze patches first
    # --------------------------------------------------------

    for patch_index, (
        row,
        col
    ) in enumerate(positions):

        image_patch = normalized_image[
            :,
            row:row + patch_size,
            col:col + patch_size
        ]

        mask_patch = mask[
            row:row + patch_size,
            col:col + patch_size
        ]

        invalid_patch = invalid_mask[
            row:row + patch_size,
            col:col + patch_size
        ]

        oil_pixels = int(
            np.sum(
                mask_patch == 1
            )
        )

        valid_pixels = int(
            np.sum(
                ~invalid_patch
            )
        )

        # ----------------------------------------------------
        # Skip completely invalid patch
        # ----------------------------------------------------

        if valid_pixels == 0:

            continue

        # ----------------------------------------------------
        # Oil patch
        # ----------------------------------------------------

        if oil_pixels >= min_oil_pixels:

            oil_candidates.append(
                (
                    patch_index,
                    row,
                    col,
                    image_patch,
                    mask_patch,
                    invalid_patch,
                    oil_pixels,
                    valid_pixels
                )
            )

        # ----------------------------------------------------
        # Background patch
        # ----------------------------------------------------

        else:

            background_candidates.append(
                (
                    patch_index,
                    row,
                    col,
                    image_patch,
                    mask_patch,
                    invalid_patch,
                    oil_pixels,
                    valid_pixels
                )
            )

    # --------------------------------------------------------
    # Keep all oil patches
    # --------------------------------------------------------

    selected_patches = []

    if keep_all_oil:

        selected_patches.extend(
            oil_candidates
        )

    else:

        selected_patches.extend(
            oil_candidates
        )

    # --------------------------------------------------------
    # Select background patches
    # --------------------------------------------------------

    if dataset_name == "oil":

        max_background_for_scene = (
            max_background
        )

    elif dataset_name == "no_oil":

        max_background_for_scene = (
            max_no_oil
        )

    elif dataset_name == "lookalike":

        max_background_for_scene = (
            max_lookalike
        )

    else:

        max_background_for_scene = 5


    if len(
        background_candidates
    ) > max_background_for_scene:

        # Deterministic sampling based on the
        # global seed + scene name.
        scene_seed = (
            RANDOM_SEED
            +
            sum(
                ord(c)
                for c in Path(
                    image_path
                ).stem
            )
        )

        scene_rng = random.Random(
            scene_seed
        )

        background_candidates = (
            scene_rng.sample(
                background_candidates,
                max_background_for_scene
            )
        )

    selected_patches.extend(
        background_candidates
    )

    # --------------------------------------------------------
    # Sort spatially
    # --------------------------------------------------------

    selected_patches.sort(
        key=lambda item: item[0]
    )

    # --------------------------------------------------------
    # Save patches
    # --------------------------------------------------------

    dataset_output_dir = (
        PATCH_DIR
        / split
        / dataset_name
    )

    saved_count = 0

    for item in selected_patches:

        (
            patch_index,
            row,
            col,
            image_patch,
            mask_patch,
            invalid_patch,
            oil_pixels,
            valid_pixels
        ) = item

        patch_name = (
            f"{Path(image_path).stem}"
            f"_r{row}_c{col}"
        )

        (
            image_output,
            mask_output,
            invalid_output
        ) = save_patch(
            image_patch,
            mask_patch,
            invalid_patch,
            dataset_output_dir,
            patch_name
        )

        patch_type = (
            "oil"
            if oil_pixels > 0
            else "background"
        )

        manifest_rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "source_image": str(
                    image_path
                ),
                "source_mask": str(
                    mask_path
                ),
                "patch_name": patch_name,
                "row": row,
                "col": col,
                "patch_size": patch_size,
                "oil_pixels": oil_pixels,
                "valid_pixels": valid_pixels,
                "oil_percentage": (
                    oil_pixels
                    / (
                        patch_size
                        * patch_size
                    )
                    * 100
                ),
                "patch_type": patch_type,
                "image_path": str(
                    image_output
                ),
                "mask_path": str(
                    mask_output
                ),
                "invalid_path": str(
                    invalid_output
                )
            }
        )

        saved_count += 1

    return {
        "total_patches": len(
            positions
        ),
        "oil_candidates": len(
            oil_candidates
        ),
        "background_candidates": len(
            background_candidates
        ),
        "saved": saved_count
    }


# ============================================================
# SPLIT SCENES
# ============================================================

def split_scenes(
    image_paths,
    train_ratio,
    seed
):
    """
    Split ORIGINAL scenes before patch generation.

    This prevents data leakage.
    """

    image_paths = list(
        image_paths
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        image_paths
    )

    split_index = int(
        len(image_paths)
        * train_ratio
    )

    train = image_paths[
        :split_index
    ]

    validation = image_paths[
        split_index:
    ]

    return (
        train,
        validation
    )


# ============================================================
# BUILD IMAGE-MASK PAIRS
# ============================================================

def build_pairs(
    image_folder,
    mask_folder
):

    images = find_tiffs(
        image_folder
    )

    pairs = []

    for image_path in images:

        mask_path = (
            Path(mask_folder)
            / image_path.name
        )

        if mask_path.exists():

            pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

    return pairs


# ============================================================
# WRITE MANIFEST
# ============================================================

def write_manifest(
    rows,
    filename
):

    manifest_path = (
        MANIFEST_DIR
        / filename
    )

    fieldnames = [
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

    with open(
        manifest_path,
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

    print(
        f"\nManifest saved:"
        f"\n{manifest_path}"
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name,
    image_folder,
    mask_folder,
    config,
    train_ratio,
    seed
):

    print("\n" + "=" * 75)

    print(
        f"PROCESSING: {dataset_name}"
    )

    print("=" * 75)

    pairs = build_pairs(
        image_folder,
        mask_folder
    )

    print(
        f"Valid image/mask pairs: "
        f"{len(pairs)}"
    )

    if not pairs:

        return []

    image_paths = [
        pair[0]
        for pair in pairs
    ]

    (
        train_images,
        validation_images
    ) = split_scenes(
        image_paths,
        train_ratio,
        seed
    )

    print(
        f"\nTrain scenes: "
        f"{len(train_images)}"
    )

    print(
        f"Validation scenes: "
        f"{len(validation_images)}"
    )

    # --------------------------------------------------------
    # Map image path → mask path
    # --------------------------------------------------------

    mask_lookup = {
        str(image): mask
        for image, mask in pairs
    }

    manifest_rows = []

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n--- TRAIN ---")

    for index, image_path in enumerate(
        train_images
    ):

        mask_path = mask_lookup[
            str(image_path)
        ]

        result = process_scene(
            image_path,
            mask_path,
            dataset_name,
            "train",
            config,
            manifest_rows
        )

        if (
            (index + 1) % 50 == 0
            or
            (index + 1) == len(train_images)
        ):

            print(
                f"Processed train scenes: "
                f"{index + 1}/"
                f"{len(train_images)}"
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n--- VALIDATION ---")

    for index, image_path in enumerate(
        validation_images
    ):

        mask_path = mask_lookup[
            str(image_path)
        ]

        result = process_scene(
            image_path,
            mask_path,
            dataset_name,
            "validation",
            config,
            manifest_rows
        )

        if (
            (index + 1) % 50 == 0
            or
            (index + 1) == len(validation_images)
        ):

            print(
                f"Processed validation scenes: "
                f"{index + 1}/"
                f"{len(validation_images)}"
            )

    return manifest_rows


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    rows
):

    print("\n" + "=" * 75)
    print(
        "PREPROCESSING SUMMARY"
    )
    print("=" * 75)

    if not rows:

        print(
            "No patches were generated."
        )

        return

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print(
        f"Total patches: "
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # Dataset counts
    # --------------------------------------------------------

    datasets = sorted(
        set(
            row["dataset"]
            for row in rows
        )
    )

    for dataset in datasets:

        dataset_rows = [
            row
            for row in rows
            if row["dataset"] == dataset
        ]

        print(
            f"\n{dataset}:"
        )

        print(
            f"  Total patches: "
            f"{len(dataset_rows):,}"
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
            f"  Oil patches: "
            f"{oil:,}"
        )

        print(
            f"  Background patches: "
            f"{background:,}"
        )

    # --------------------------------------------------------
    # Split counts
    # --------------------------------------------------------

    print("\nSplit:")

    for split in [
        "train",
        "validation"
    ]:

        split_rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        print(
            f"  {split}: "
            f"{len(split_rows):,}"
        )

    # --------------------------------------------------------
    # Patch size
    # --------------------------------------------------------

    patch_sizes = sorted(
        set(
            row["patch_size"]
            for row in rows
        )
    )

    print(
        f"\nPatch sizes: "
        f"{patch_sizes}"
    )

    # --------------------------------------------------------
    # Oil percentage
    # --------------------------------------------------------

    oil_percentages = np.asarray(
        [
            float(row["oil_percentage"])
            for row in rows
            if row["patch_type"] == "oil"
        ]
    )

    if len(oil_percentages) > 0:

        print(
            "\nOil percentage within "
            "oil-containing patches:"
        )

        for percentile in [
            0,
            25,
            50,
            75,
            95,
            99,
            100
        ]:

            value = np.percentile(
                oil_percentages,
                percentile
            )

            print(
                f"  P{percentile}: "
                f"{value:.4f}%"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "             OIL SPILL PREPROCESSING"
    )
    print("=" * 75)

    config = load_config()

    create_directories()

    split_config = config.get(
        "split",
        {}
    )

    train_ratio = split_config.get(
        "train",
        0.8
    )

    seed = split_config.get(
        "seed",
        RANDOM_SEED
    )

    all_rows = []

    dataset = config[
        "dataset"
    ]

    # ========================================================
    # PART 1 - OIL
    # ========================================================

    oil_rows = process_dataset(
        "oil",
        dataset["oil"]["images"],
        dataset["oil"]["masks"],
        config,
        train_ratio,
        seed
    )

    all_rows.extend(
        oil_rows
    )

    # ========================================================
    # PART 2 - NO OIL
    # ========================================================

    no_oil_rows = process_dataset(
        "no_oil",
        dataset["no_oil"]["images"],
        dataset["no_oil"]["masks"],
        config,
        train_ratio,
        seed
    )

    all_rows.extend(
        no_oil_rows
    )

    # ========================================================
    # PART 2 - LOOKALIKE
    # ========================================================

    lookalike_rows = process_dataset(
        "lookalike",
        dataset["lookalike"]["images"],
        dataset["lookalike"]["masks"],
        config,
        train_ratio,
        seed
    )

    all_rows.extend(
        lookalike_rows
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    write_manifest(
        all_rows,
        "patch_manifest.csv"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        all_rows
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "PREPROCESSING COMPLETE"
    )
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()