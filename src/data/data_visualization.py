from pathlib import Path

import yaml
import rasterio
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ============================================================
# CONFIG
# ============================================================

def load_config():
    """
    Load dataset paths and project configuration
    from config.yaml.
    """

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# TIFF FINDER
# ============================================================

def find_tiffs(folder):
    """
    Find all TIFF files recursively inside a folder.
    """

    folder = Path(folder)

    return sorted(
        list(folder.rglob("*.tif"))
        + list(folder.rglob("*.tiff"))
    )


# ============================================================
# DISPLAY NORMALIZATION
# ============================================================

def normalize_for_display(image, lower=2, upper=98):
    """
    Normalize SAR image only for visualization.

    IMPORTANT:
    This normalization is NOT used for model training.
    """

    low = np.percentile(image, lower)
    high = np.percentile(image, upper)

    image = np.clip(
        image,
        low,
        high
    )

    if high > low:

        image = (
            image - low
        ) / (
            high - low
        )

    else:

        image = np.zeros_like(image)

    return image


# ============================================================
# LOAD IMAGE + MASK
# ============================================================

def load_sample(image_path, mask_path):
    """
    Load a two-band Sentinel-1 image
    and its corresponding mask.
    """

    with rasterio.open(image_path) as src:

        image = src.read()

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    return image, mask


# ============================================================
# OIL PERCENTAGE
# ============================================================

def get_oil_percentage(mask_path):
    """
    Calculate percentage of pixels labelled as oil.
    """

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    oil_pixels = np.sum(mask == 1)

    total_pixels = mask.size

    percentage = (
        oil_pixels
        / total_pixels
        * 100
    )

    return percentage


# ============================================================
# VISUALIZE SAMPLE
# ============================================================

def visualize_sample(
    image_path,
    mask_path,
    dataset_name,
    oil_percentage=None
):
    """
    Display:

    1. VV
    2. VH
    3. Ground Truth Mask
    4. Oil Overlay
    """

    image, mask = load_sample(
        image_path,
        mask_path
    )

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image.shape[0] < 2:

        print(
            f"ERROR: Image has fewer than 2 bands: "
            f"{image_path}"
        )

        return

    # --------------------------------------------------------
    # Extract bands
    # --------------------------------------------------------

    band_1 = image[0]
    band_2 = image[1]

    # At this stage we display them as Band 1 / Band 2.
    # We will verify VV/VH ordering before preprocessing.
    vv = band_1
    vh = band_2

    # --------------------------------------------------------
    # Display normalization
    # --------------------------------------------------------

    vv_display = normalize_for_display(vv)
    vh_display = normalize_for_display(vh)

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay_rgb = np.stack(
        [
            vv_display,
            vv_display,
            vv_display
        ],
        axis=-1
    )

    # Highlight oil pixels
    overlay_rgb[mask == 1] = [
        1,
        0,
        0
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # --------------------------------------------------------
    # VV
    # --------------------------------------------------------

    axes[0].imshow(
        vv_display,
        cmap="gray"
    )

    axes[0].set_title(
        "Band 1 / VV"
    )

    # --------------------------------------------------------
    # VH
    # --------------------------------------------------------

    axes[1].imshow(
        vh_display,
        cmap="gray"
    )

    axes[1].set_title(
        "Band 2 / VH"
    )

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[2].imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Ground Truth Mask"
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[3].imshow(
        overlay_rgb
    )

    axes[3].set_title(
        "Oil Overlay"
    )

    # --------------------------------------------------------
    # Remove axes
    # --------------------------------------------------------

    for ax in axes:

        ax.axis("off")

    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    if oil_percentage is not None:

        title = (
            f"{dataset_name} | "
            f"{image_path.name} | "
            f"Oil: {oil_percentage:.4f}%"
        )

    else:

        title = (
            f"{dataset_name} | "
            f"{image_path.name}"
        )

    fig.suptitle(
        title,
        fontsize=14
    )

    plt.tight_layout()

    plt.show()

    plt.close(fig)


# ============================================================
# OIL SPILL SIZE VISUALIZATION
# ============================================================

def visualize_oil_by_size(
    image_folder,
    mask_folder,
    num_each=2
):
    """
    Select oil images based on the percentage
    of pixels labelled as oil.

    Categories:

    SMALL
    MEDIUM
    LARGE
    """

    images = find_tiffs(
        image_folder
    )

    print("\n" + "=" * 70)
    print("SELECTING OIL SAMPLES BY SPILL SIZE")
    print("=" * 70)

    print(
        f"Total oil images: {len(images)}"
    )

    samples = []

    # --------------------------------------------------------
    # Calculate oil percentage for every image
    # --------------------------------------------------------

    for i, image_path in enumerate(images):

        mask_path = (
            Path(mask_folder)
            / image_path.name
        )

        if not mask_path.exists():

            print(
                f"WARNING: Mask not found for "
                f"{image_path.name}"
            )

            continue

        try:

            percentage = get_oil_percentage(
                mask_path
            )

            samples.append(
                (
                    percentage,
                    image_path
                )
            )

        except Exception as error:

            print(
                f"ERROR processing "
                f"{image_path.name}: "
                f"{error}"
            )

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(images)}"
            )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not samples:

        print(
            "No valid oil samples found."
        )

        return

    # --------------------------------------------------------
    # Sort by oil percentage
    # --------------------------------------------------------

    samples.sort(
        key=lambda item: item[0]
    )

    total = len(samples)

    # --------------------------------------------------------
    # SMALL
    # --------------------------------------------------------

    small = samples[
        :num_each
    ]

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    middle_start = (
        total // 2
        - num_each // 2
    )

    medium = samples[
        middle_start:
        middle_start + num_each
    ]

    # --------------------------------------------------------
    # LARGE
    # --------------------------------------------------------

    large = samples[
        -num_each:
    ]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    selected = (
        [
            ("SMALL", sample)
            for sample in small
        ]
        +
        [
            ("MEDIUM", sample)
            for sample in medium
        ]
        +
        [
            ("LARGE", sample)
            for sample in large
        ]
    )

    # --------------------------------------------------------
    # Print selected samples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTED OIL SAMPLES")
    print("=" * 70)

    for category, (
        percentage,
        image_path
    ) in selected:

        print(
            f"{category:8} | "
            f"{image_path.name:15} | "
            f"Oil pixels: "
            f"{percentage:.4f}%"
        )

    # --------------------------------------------------------
    # Visualize
    # --------------------------------------------------------

    for category, (
        percentage,
        image_path
    ) in selected:

        mask_path = (
            Path(mask_folder)
            / image_path.name
        )

        visualize_sample(
            image_path,
            mask_path,
            f"OIL - {category}",
            oil_percentage=percentage
        )


# ============================================================
# RANDOM / EVENLY SPACED DATASET VISUALIZATION
# ============================================================

def visualize_dataset(
    dataset_name,
    image_folder,
    mask_folder,
    num_samples=2
):
    """
    Visualize evenly spaced samples.

    Used for:

    - No Oil
    - Lookalike
    """

    images = find_tiffs(
        image_folder
    )

    print("\n" + "=" * 70)
    print(
        f"VISUALIZING: {dataset_name}"
    )
    print("=" * 70)

    print(
        f"Total images available: "
        f"{len(images)}"
    )

    if not images:

        print(
            "No TIFF images found."
        )

        return

    # --------------------------------------------------------
    # Select images
    # --------------------------------------------------------

    if len(images) <= num_samples:

        selected = images

    else:

        indexes = np.linspace(
            0,
            len(images) - 1,
            num_samples,
            dtype=int
        )

        selected = [
            images[index]
            for index in indexes
        ]

    # --------------------------------------------------------
    # Visualize
    # --------------------------------------------------------

    for image_path in selected:

        mask_path = (
            Path(mask_folder)
            / image_path.name
        )

        if not mask_path.exists():

            print(
                f"Mask not found: "
                f"{image_path.name}"
            )

            continue

        visualize_sample(
            image_path,
            mask_path,
            dataset_name
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "        OIL SPILL DATASET VISUALIZATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config()

    dataset = config["dataset"]

    # ========================================================
    # PART 1 - OIL
    # ========================================================

    visualize_oil_by_size(
        dataset["oil"]["images"],
        dataset["oil"]["masks"],
        num_each=2
    )

    # ========================================================
    # PART 2 - NO OIL
    # ========================================================

    visualize_dataset(
        "PART 2 - NO OIL",
        dataset["no_oil"]["images"],
        dataset["no_oil"]["masks"],
        num_samples=2
    )

    # ========================================================
    # PART 2 - LOOKALIKE
    # ========================================================

    visualize_dataset(
        "PART 2 - LOOKALIKE",
        dataset["lookalike"]["images"],
        dataset["lookalike"]["masks"],
        num_samples=2
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "VISUALIZATION COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()