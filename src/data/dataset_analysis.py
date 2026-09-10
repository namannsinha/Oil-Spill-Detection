from pathlib import Path

import yaml
import rasterio
import numpy as np
from scipy import ndimage


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# Number of pixels sampled from each image for pixel-level
# intensity statistics.
PIXELS_PER_IMAGE = 2000

# Fixed seed makes the analysis reproducible.
RANDOM_SEED = 42

# 8-connectivity:
#
# 1 1 1
# 1 1 1
# 1 1 1
#
# This connects diagonal pixels as part of the same region.
CONNECTIVITY_8 = np.ones(
    (3, 3),
    dtype=np.uint8
)


# ============================================================
# CONFIG
# ============================================================

def load_config():
    """
    Load configuration from config.yaml.
    """

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# TIFF FINDER
# ============================================================

def find_tiffs(folder):
    """
    Recursively find all TIFF files.
    """

    folder = Path(folder)

    return sorted(
        list(folder.rglob("*.tif"))
        + list(folder.rglob("*.tiff"))
    )


# ============================================================
# PERCENTILE PRINTING
# ============================================================

def print_percentiles(
    name,
    values,
    decimals=4
):
    """
    Print useful percentiles for a numeric array.
    """

    values = np.asarray(values)

    if values.size == 0:

        print(
            f"\n{name}: No data available."
        )

        return

    print(f"\n{name}")

    percentiles = [
        0.1,
        1,
        2,
        5,
        25,
        50,
        75,
        95,
        98,
        99,
        99.9
    ]

    for percentile in percentiles:

        value = np.percentile(
            values,
            percentile
        )

        print(
            f"  P{percentile:g}: "
            f"{value:.{decimals}f}"
        )


# ============================================================
# PIXEL SAMPLING
# ============================================================

def sample_pixels(
    image,
    number_of_pixels,
    rng
):
    """
    Randomly sample pixels from an image band.

    Only finite values are considered.
    """

    flat = image.reshape(-1)

    finite_indices = np.flatnonzero(
        np.isfinite(flat)
    )

    if len(finite_indices) == 0:

        return np.array(
            [],
            dtype=np.float32
        )

    sample_size = min(
        number_of_pixels,
        len(finite_indices)
    )

    selected_indices = rng.choice(
        finite_indices,
        size=sample_size,
        replace=False
    )

    return flat[
        selected_indices
    ].astype(np.float32)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path,
    rng
):
    """
    Analyze a two-band Sentinel-1 image.
    """

    with rasterio.open(image_path) as src:

        image = src.read()

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if image.shape[0] < 2:

            raise ValueError(
                "Image contains fewer than 2 bands."
            )

        band1 = image[0]
        band2 = image[1]

        # ----------------------------------------------------
        # Image-level statistics
        # ----------------------------------------------------

        band1_finite = band1[
            np.isfinite(band1)
        ]

        band2_finite = band2[
            np.isfinite(band2)
        ]

        stats = {

            "band1_mean": float(
                np.mean(band1_finite)
            ),

            "band1_std": float(
                np.std(band1_finite)
            ),

            "band1_min": float(
                np.min(band1_finite)
            ),

            "band1_max": float(
                np.max(band1_finite)
            ),

            "band2_mean": float(
                np.mean(band2_finite)
            ),

            "band2_std": float(
                np.std(band2_finite)
            ),

            "band2_min": float(
                np.min(band2_finite)
            ),

            "band2_max": float(
                np.max(band2_finite)
            ),

            "nan_pixels": int(
                np.isnan(band1).sum()
                +
                np.isnan(band2).sum()
            ),

            "inf_pixels": int(
                np.isinf(band1).sum()
                +
                np.isinf(band2).sum()
            ),

            "shape": image.shape,

            "crs": str(src.crs),

            "transform": src.transform
        }

        # ----------------------------------------------------
        # Pixel samples
        # ----------------------------------------------------

        band1_sample = sample_pixels(
            band1,
            PIXELS_PER_IMAGE,
            rng
        )

        band2_sample = sample_pixels(
            band2,
            PIXELS_PER_IMAGE,
            rng
        )

    return (
        stats,
        band1_sample,
        band2_sample
    )


# ============================================================
# MASK ANALYSIS
# ============================================================

def analyze_mask(mask_path):
    """
    Analyze an oil mask using 8-connected components.

    Returns:
        - oil pixel count
        - oil percentage
        - number of connected regions
        - component areas
        - bounding box widths
        - bounding box heights
        - bounding box areas
    """

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    # --------------------------------------------------------
    # Basic mask statistics
    # --------------------------------------------------------

    total_pixels = mask.size

    oil_mask = mask == 1

    oil_pixels = int(
        np.sum(oil_mask)
    )

    oil_percentage = (
        oil_pixels
        / total_pixels
        * 100
    )

    # --------------------------------------------------------
    # No oil
    # --------------------------------------------------------

    if oil_pixels == 0:

        return {
            "total_pixels": total_pixels,
            "oil_pixels": 0,
            "oil_percentage": 0.0,
            "num_components": 0,
            "component_sizes": np.array(
                [],
                dtype=np.int64
            ),
            "bbox_widths": np.array(
                [],
                dtype=np.int64
            ),
            "bbox_heights": np.array(
                [],
                dtype=np.int64
            ),
            "bbox_areas": np.array(
                [],
                dtype=np.int64
            )
        }

    # --------------------------------------------------------
    # 8-connected component labeling
    # --------------------------------------------------------

    labeled, num_components = ndimage.label(
        oil_mask,
        structure=CONNECTIVITY_8
    )

    # --------------------------------------------------------
    # Component sizes
    # --------------------------------------------------------

    component_sizes = np.bincount(
        labeled.ravel()
    )[1:]

    # --------------------------------------------------------
    # Bounding boxes
    # --------------------------------------------------------

    objects = ndimage.find_objects(
        labeled
    )

    bbox_widths = []
    bbox_heights = []
    bbox_areas = []

    for obj in objects:

        if obj is None:
            continue

        row_slice, col_slice = obj

        height = (
            row_slice.stop
            - row_slice.start
        )

        width = (
            col_slice.stop
            - col_slice.start
        )

        area = (
            height
            * width
        )

        bbox_widths.append(width)
        bbox_heights.append(height)
        bbox_areas.append(area)

    return {
        "total_pixels": total_pixels,
        "oil_pixels": oil_pixels,
        "oil_percentage": float(
            oil_percentage
        ),
        "num_components": int(
            num_components
        ),
        "component_sizes": np.asarray(
            component_sizes,
            dtype=np.int64
        ),
        "bbox_widths": np.asarray(
            bbox_widths,
            dtype=np.int64
        ),
        "bbox_heights": np.asarray(
            bbox_heights,
            dtype=np.int64
        ),
        "bbox_areas": np.asarray(
            bbox_areas,
            dtype=np.int64
        )
    }


# ============================================================
# IMAGE DATASET ANALYSIS
# ============================================================

def analyze_dataset(
    name,
    image_folder,
    mask_folder=None
):
    """
    Perform complete analysis for one dataset.
    """

    images = find_tiffs(
        image_folder
    )

    print("\n" + "=" * 75)
    print(
        f"ANALYZING: {name}"
    )
    print("=" * 75)

    print(
        f"Images found: {len(images)}"
    )

    if not images:

        print(
            "No images found."
        )

        return

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    band1_means = []
    band1_stds = []
    band1_mins = []
    band1_maxs = []

    band2_means = []
    band2_stds = []
    band2_mins = []
    band2_maxs = []

    band1_pixel_samples = []
    band2_pixel_samples = []

    oil_percentages = []

    component_counts = []

    component_sizes_all = []

    bbox_widths_all = []
    bbox_heights_all = []
    bbox_areas_all = []

    problematic_images = []

    total_pixels = 0
    total_oil_pixels = 0

    images_with_oil = 0

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_path in enumerate(images):

        try:

            (
                image_stats,
                band1_sample,
                band2_sample
            ) = analyze_image(
                image_path,
                rng
            )

            # ------------------------------------------------
            # Image statistics
            # ------------------------------------------------

            band1_means.append(
                image_stats["band1_mean"]
            )

            band1_stds.append(
                image_stats["band1_std"]
            )

            band1_mins.append(
                image_stats["band1_min"]
            )

            band1_maxs.append(
                image_stats["band1_max"]
            )

            band2_means.append(
                image_stats["band2_mean"]
            )

            band2_stds.append(
                image_stats["band2_std"]
            )

            band2_mins.append(
                image_stats["band2_min"]
            )

            band2_maxs.append(
                image_stats["band2_max"]
            )

            # ------------------------------------------------
            # Pixel samples
            # ------------------------------------------------

            if len(band1_sample) > 0:

                band1_pixel_samples.append(
                    band1_sample
                )

            if len(band2_sample) > 0:

                band2_pixel_samples.append(
                    band2_sample
                )

            # ------------------------------------------------
            # Data quality
            # ------------------------------------------------

            if (
                image_stats["nan_pixels"] > 0
                or image_stats["inf_pixels"] > 0
            ):

                problematic_images.append(
                    str(image_path)
                )

            # ------------------------------------------------
            # Mask analysis
            # ------------------------------------------------

            if mask_folder is not None:

                mask_path = (
                    Path(mask_folder)
                    / image_path.name
                )

                if not mask_path.exists():

                    problematic_images.append(
                        f"Missing mask: "
                        f"{image_path.name}"
                    )

                else:

                    mask_stats = analyze_mask(
                        mask_path
                    )

                    total_pixels += (
                        mask_stats[
                            "total_pixels"
                        ]
                    )

                    total_oil_pixels += (
                        mask_stats[
                            "oil_pixels"
                        ]
                    )

                    oil_percentages.append(
                        mask_stats[
                            "oil_percentage"
                        ]
                    )

                    component_counts.append(
                        mask_stats[
                            "num_components"
                        ]
                    )

                    if (
                        mask_stats[
                            "oil_pixels"
                        ] > 0
                    ):

                        images_with_oil += 1

                    # ----------------------------------------
                    # Components
                    # ----------------------------------------

                    if len(
                        mask_stats[
                            "component_sizes"
                        ]
                    ) > 0:

                        component_sizes_all.extend(
                            mask_stats[
                                "component_sizes"
                            ]
                        )

                    # ----------------------------------------
                    # Bounding boxes
                    # ----------------------------------------

                    if len(
                        mask_stats[
                            "bbox_widths"
                        ]
                    ) > 0:

                        bbox_widths_all.extend(
                            mask_stats[
                                "bbox_widths"
                            ]
                        )

                    if len(
                        mask_stats[
                            "bbox_heights"
                        ]
                    ) > 0:

                        bbox_heights_all.extend(
                            mask_stats[
                                "bbox_heights"
                            ]
                        )

                    if len(
                        mask_stats[
                            "bbox_areas"
                        ]
                    ) > 0:

                        bbox_areas_all.extend(
                            mask_stats[
                                "bbox_areas"
                            ]
                        )

        except Exception as error:

            problematic_images.append(
                f"{image_path} -> {error}"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 100 == 0
            or
            (index + 1) == len(images)
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(images)}"
            )

    # ========================================================
    # IMAGE STATISTICS
    # ========================================================

    print("\n" + "-" * 75)
    print("IMAGE-LEVEL STATISTICS")
    print("-" * 75)

    # --------------------------------------------------------
    # Band 1
    # --------------------------------------------------------

    print_percentiles(
        "Band 1 Mean",
        band1_means
    )

    print_percentiles(
        "Band 1 Std",
        band1_stds
    )

    print(
        f"\nBand 1 global observed minimum: "
        f"{min(band1_mins):.4f}"
    )

    print(
        f"Band 1 global observed maximum: "
        f"{max(band1_maxs):.4f}"
    )

    # --------------------------------------------------------
    # Band 2
    # --------------------------------------------------------

    print_percentiles(
        "Band 2 Mean",
        band2_means
    )

    print_percentiles(
        "Band 2 Std",
        band2_stds
    )

    print(
        f"\nBand 2 global observed minimum: "
        f"{min(band2_mins):.4f}"
    )

    print(
        f"Band 2 global observed maximum: "
        f"{max(band2_maxs):.4f}"
    )

    # ========================================================
    # PIXEL-LEVEL STATISTICS
    # ========================================================

    print("\n" + "-" * 75)
    print(
        "PIXEL-LEVEL SAR INTENSITY STATISTICS"
    )
    print("-" * 75)

    if band1_pixel_samples:

        band1_pixels = np.concatenate(
            band1_pixel_samples
        )

        print_percentiles(
            "Band 1 Pixel Distribution",
            band1_pixels
        )

    else:

        band1_pixels = np.array(
            [],
            dtype=np.float32
        )

    if band2_pixel_samples:

        band2_pixels = np.concatenate(
            band2_pixel_samples
        )

        print_percentiles(
            "Band 2 Pixel Distribution",
            band2_pixels
        )

    else:

        band2_pixels = np.array(
            [],
            dtype=np.float32
        )

    # ========================================================
    # MASK / OIL STATISTICS
    # ========================================================

    if mask_folder is not None:

        print("\n" + "-" * 75)
        print("OIL / MASK STATISTICS")
        print("-" * 75)

        print(
            f"Images containing oil: "
            f"{images_with_oil}/{len(images)} "
            f"("
            f"{images_with_oil / len(images) * 100:.2f}"
            f"%)"
        )

        print(
            f"\nTotal mask pixels: "
            f"{total_pixels:,}"
        )

        print(
            f"Total oil pixels: "
            f"{total_oil_pixels:,}"
        )

        if total_pixels > 0:

            oil_ratio = (
                total_oil_pixels
                / total_pixels
                * 100
            )

            print(
                f"Overall oil pixels: "
                f"{oil_ratio:.6f}%"
            )

            print(
                f"Overall background: "
                f"{100 - oil_ratio:.6f}%"
            )

        # ----------------------------------------------------
        # Oil percentage per image
        # ----------------------------------------------------

        if oil_percentages:

            print_percentiles(
                "Oil Percentage Per Image",
                oil_percentages
            )

        # ----------------------------------------------------
        # Components per image
        # ----------------------------------------------------

        if component_counts:

            print_percentiles(
                "Connected Components Per Image",
                component_counts,
                decimals=2
            )

        # ====================================================
        # COMPONENT SIZE
        # ====================================================

        if component_sizes_all:

            component_sizes = np.asarray(
                component_sizes_all
            )

            print("\n" + "-" * 75)
            print(
                "8-CONNECTED OIL COMPONENT SIZES"
            )
            print("-" * 75)

            print(
                f"Total connected components: "
                f"{len(component_sizes):,}"
            )

            print_percentiles(
                "Component Pixel Area",
                component_sizes,
                decimals=2
            )

            print(
                f"\nSmallest component: "
                f"{component_sizes.min():,} pixels"
            )

            print(
                f"Largest component: "
                f"{component_sizes.max():,} pixels"
            )

        # ====================================================
        # BOUNDING BOX WIDTH
        # ====================================================

        if bbox_widths_all:

            bbox_widths = np.asarray(
                bbox_widths_all
            )

            print("\n" + "-" * 75)
            print(
                "OIL REGION BOUNDING BOX WIDTH"
            )
            print("-" * 75)

            print_percentiles(
                "Bounding Box Width (pixels)",
                bbox_widths,
                decimals=2
            )

            print(
                f"\nMaximum width: "
                f"{bbox_widths.max():,} pixels"
            )

        # ====================================================
        # BOUNDING BOX HEIGHT
        # ====================================================

        if bbox_heights_all:

            bbox_heights = np.asarray(
                bbox_heights_all
            )

            print("\n" + "-" * 75)
            print(
                "OIL REGION BOUNDING BOX HEIGHT"
            )
            print("-" * 75)

            print_percentiles(
                "Bounding Box Height (pixels)",
                bbox_heights,
                decimals=2
            )

            print(
                f"\nMaximum height: "
                f"{bbox_heights.max():,} pixels"
            )

        # ====================================================
        # BOUNDING BOX AREA
        # ====================================================

        if bbox_areas_all:

            bbox_areas = np.asarray(
                bbox_areas_all
            )

            print("\n" + "-" * 75)
            print(
                "OIL REGION BOUNDING BOX AREA"
            )
            print("-" * 75)

            print_percentiles(
                "Bounding Box Area (pixels)",
                bbox_areas,
                decimals=2
            )

            print(
                f"\nLargest bounding box: "
                f"{bbox_areas.max():,} pixels"
            )

        # ====================================================
        # PATCH-SIZE CHECK
        # ====================================================

        if bbox_widths_all and bbox_heights_all:

            bbox_widths = np.asarray(
                bbox_widths_all
            )

            bbox_heights = np.asarray(
                bbox_heights_all
            )

            print("\n" + "-" * 75)
            print(
                "PATCH SIZE REFERENCE"
            )
            print("-" * 75)

            for patch_size in [
                256,
                512
            ]:

                width_covered = np.mean(
                    bbox_widths <= patch_size
                ) * 100

                height_covered = np.mean(
                    bbox_heights <= patch_size
                ) * 100

                print(
                    f"\n{patch_size}x{patch_size} patch:"
                )

                print(
                    f"  Components with "
                    f"width <= {patch_size}: "
                    f"{width_covered:.2f}%"
                )

                print(
                    f"  Components with "
                    f"height <= {patch_size}: "
                    f"{height_covered:.2f}%"
                )

        # ====================================================
        # VERY LARGE SPILLS
        # ====================================================

        if oil_percentages:

            oil_percentages_array = np.asarray(
                oil_percentages
            )

            print("\n" + "-" * 75)
            print(
                "LARGE OIL SCENE ANALYSIS"
            )
            print("-" * 75)

            thresholds = [
                1,
                5,
                10,
                25,
                50
            ]

            for threshold in thresholds:

                count = np.sum(
                    oil_percentages_array >= threshold
                )

                percentage = (
                    count
                    / len(oil_percentages_array)
                    * 100
                )

                print(
                    f"Images with >= "
                    f"{threshold}% oil: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print("\n" + "-" * 75)
    print("DATA QUALITY")
    print("-" * 75)

    print(
        f"Problematic images: "
        f"{len(problematic_images)}"
    )

    if problematic_images:

        print("\nExamples:")

        for problem in problematic_images[:10]:

            print(
                f"  {problem}"
            )

    else:

        print(
            "No NaN, Inf, or processing errors detected."
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "images": len(images),
        "images_with_oil": images_with_oil,
        "oil_percentages": oil_percentages,
        "component_counts": component_counts,
        "component_sizes": component_sizes_all,
        "bbox_widths": bbox_widths_all,
        "bbox_heights": bbox_heights_all,
        "bbox_areas": bbox_areas_all,
        "band1_pixels": band1_pixels,
        "band2_pixels": band2_pixels,
        "problematic_images": problematic_images
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "             OIL SPILL DATASET ANALYSIS"
    )
    print("=" * 75)

    print(
        f"\nPixel samples per image: "
        f"{PIXELS_PER_IMAGE:,}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    config = load_config()

    dataset = config["dataset"]

    # ========================================================
    # PART 1 - OIL
    # ========================================================

    oil_results = analyze_dataset(
        "PART 1 - OIL",
        dataset["oil"]["images"],
        dataset["oil"]["masks"]
    )

    # ========================================================
    # PART 2 - NO OIL
    # ========================================================

    no_oil_results = analyze_dataset(
        "PART 2 - NO OIL",
        dataset["no_oil"]["images"],
        dataset["no_oil"]["masks"]
    )

    # ========================================================
    # PART 2 - LOOKALIKE
    # ========================================================

    lookalike_results = analyze_dataset(
        "PART 2 - LOOKALIKE",
        dataset["lookalike"]["images"],
        dataset["lookalike"]["masks"]
    )

    # ========================================================
    # COMBINED PIXEL STATISTICS
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "COMBINED PIXEL-LEVEL STATISTICS"
    )
    print("=" * 75)

    all_band1 = []
    all_band2 = []

    for result in [
        oil_results,
        no_oil_results,
        lookalike_results
    ]:

        if result is None:
            continue

        if len(
            result["band1_pixels"]
        ) > 0:

            all_band1.append(
                result["band1_pixels"]
            )

        if len(
            result["band2_pixels"]
        ) > 0:

            all_band2.append(
                result["band2_pixels"]
            )

    if all_band1:

        combined_band1 = np.concatenate(
            all_band1
        )

        print_percentiles(
            "COMBINED Band 1 Pixel Distribution",
            combined_band1
        )

    if all_band2:

        combined_band2 = np.concatenate(
            all_band2
        )

        print_percentiles(
            "COMBINED Band 2 Pixel Distribution",
            combined_band2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "DATASET ANALYSIS COMPLETE"
    )
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()