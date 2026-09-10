from pathlib import Path

import yaml
import rasterio
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ============================================================
# SETTINGS
# ============================================================

# Number of valid pixels sampled from each image.
PIXELS_PER_IMAGE = 5000

RANDOM_SEED = 42


# ============================================================
# CONFIG
# ============================================================

def load_config():

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# TIFF FINDER
# ============================================================

def find_tiffs(folder):

    folder = Path(folder)

    return sorted(
        list(folder.rglob("*.tif"))
        + list(folder.rglob("*.tiff"))
    )


# ============================================================
# SAMPLE VALID PIXELS
# ============================================================

def sample_valid_pixels(
    data,
    rng,
    number_of_pixels
):
    """
    Return random pixels that are:

    - finite
    - not equal to zero
    """

    data = data.reshape(-1)

    valid = (
        np.isfinite(data)
        &
        (data != 0)
    )

    valid_pixels = data[valid]

    if len(valid_pixels) == 0:

        return np.array(
            [],
            dtype=np.float32
        )

    sample_size = min(
        number_of_pixels,
        len(valid_pixels)
    )

    indices = rng.choice(
        len(valid_pixels),
        size=sample_size,
        replace=False
    )

    return valid_pixels[
        indices
    ].astype(np.float32)


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset(
    name,
    image_folder,
    rng
):

    images = find_tiffs(
        image_folder
    )

    print("\n" + "=" * 70)
    print(
        f"ANALYZING: {name}"
    )
    print("=" * 70)

    print(
        f"Images: {len(images)}"
    )

    vv_samples = []
    vh_samples = []

    for index, image_path in enumerate(images):

        try:

            with rasterio.open(image_path) as src:

                vv = src.read(1)
                vh = src.read(2)

            vv_valid = sample_valid_pixels(
                vv,
                rng,
                PIXELS_PER_IMAGE
            )

            vh_valid = sample_valid_pixels(
                vh,
                rng,
                PIXELS_PER_IMAGE
            )

            if len(vv_valid) > 0:

                vv_samples.append(
                    vv_valid
                )

            if len(vh_valid) > 0:

                vh_samples.append(
                    vh_valid
                )

        except Exception as error:

            print(
                f"ERROR: {image_path}"
            )

            print(error)

        if (
            (index + 1) % 100 == 0
            or
            (index + 1) == len(images)
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(images)}"
            )

    return (
        np.concatenate(vv_samples),
        np.concatenate(vh_samples)
    )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    values
):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Valid sampled pixels: "
        f"{len(values):,}"
    )

    print(
        f"Minimum: "
        f"{np.min(values):.6f}"
    )

    print(
        f"Maximum: "
        f"{np.max(values):.6f}"
    )

    percentiles = [
        0.1,
        1,
        2,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        98,
        99,
        99.5,
        99.9
    ]

    for percentile in percentiles:

        value = np.percentile(
            values,
            percentile
        )

        print(
            f"P{percentile:g}: "
            f"{value:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "       VALID SAR PIXEL DISTRIBUTION"
    )
    print("=" * 70)

    print(
        "\nZero pixels are excluded."
    )

    print(
        f"Pixels sampled per image: "
        f"{PIXELS_PER_IMAGE:,}"
    )

    config = load_config()

    dataset = config["dataset"]

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # ========================================================
    # OIL
    # ========================================================

    oil_vv, oil_vh = analyze_dataset(
        "PART 1 - OIL",
        dataset["oil"]["images"],
        rng
    )

    # ========================================================
    # NO OIL
    # ========================================================

    no_oil_vv, no_oil_vh = analyze_dataset(
        "PART 2 - NO OIL",
        dataset["no_oil"]["images"],
        rng
    )

    # ========================================================
    # LOOKALIKE
    # ========================================================

    lookalike_vv, lookalike_vh = analyze_dataset(
        "PART 2 - LOOKALIKE",
        dataset["lookalike"]["images"],
        rng
    )

    # ========================================================
    # COMBINE
    # ========================================================

    combined_vv = np.concatenate(
        [
            oil_vv,
            no_oil_vv,
            lookalike_vv
        ]
    )

    combined_vh = np.concatenate(
        [
            oil_vh,
            no_oil_vh,
            lookalike_vh
        ]
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print(
        "COMBINED VALID PIXEL DISTRIBUTION"
    )
    print("=" * 70)

    print_distribution(
        "VV - ZERO VALUES EXCLUDED",
        combined_vv
    )

    print_distribution(
        "VH - ZERO VALUES EXCLUDED",
        combined_vh
    )

    # ========================================================
    # SUGGESTED CLIPPING
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "ROBUST NORMALIZATION CANDIDATES"
    )
    print("=" * 70)

    vv_low = np.percentile(
        combined_vv,
        1
    )

    vv_high = np.percentile(
        combined_vv,
        99
    )

    vh_low = np.percentile(
        combined_vh,
        1
    )

    vh_high = np.percentile(
        combined_vh,
        99
    )

    print(
        "\nVV:"
    )

    print(
        f"  1st percentile : "
        f"{vv_low:.6f}"
    )

    print(
        f"  99th percentile: "
        f"{vv_high:.6f}"
    )

    print(
        "\nVH:"
    )

    print(
        f"  1st percentile : "
        f"{vh_low:.6f}"
    )

    print(
        f"  99th percentile: "
        f"{vh_high:.6f}"
    )

    print("\n" + "=" * 70)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()