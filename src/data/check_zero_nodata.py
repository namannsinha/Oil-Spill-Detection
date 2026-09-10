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

# Width of the border used to check whether zero pixels
# are concentrated near image edges.
EDGE_WIDTH = 32


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
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(path):

    with rasterio.open(path) as src:

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        nodata_values = src.nodatavals

        # ----------------------------------------------------
        # Read both bands
        # ----------------------------------------------------

        vv = src.read(1)
        vh = src.read(2)

        height, width = vv.shape

        total_pixels = height * width

        # ----------------------------------------------------
        # Zero pixels
        # ----------------------------------------------------

        vv_zero = vv == 0
        vh_zero = vh == 0

        vv_zero_count = int(
            np.sum(vv_zero)
        )

        vh_zero_count = int(
            np.sum(vh_zero)
        )

        vv_zero_percentage = (
            vv_zero_count
            / total_pixels
            * 100
        )

        vh_zero_percentage = (
            vh_zero_count
            / total_pixels
            * 100
        )

        # ----------------------------------------------------
        # Edge mask
        # ----------------------------------------------------

        edge_mask = np.zeros(
            (height, width),
            dtype=bool
        )

        edge_mask[:EDGE_WIDTH, :] = True
        edge_mask[-EDGE_WIDTH:, :] = True
        edge_mask[:, :EDGE_WIDTH] = True
        edge_mask[:, -EDGE_WIDTH:] = True

        interior_mask = ~edge_mask

        # ----------------------------------------------------
        # Edge zero statistics
        # ----------------------------------------------------

        edge_pixels = np.sum(
            edge_mask
        )

        interior_pixels = np.sum(
            interior_mask
        )

        vv_edge_zero = np.sum(
            vv_zero & edge_mask
        )

        vh_edge_zero = np.sum(
            vh_zero & edge_mask
        )

        vv_interior_zero = np.sum(
            vv_zero & interior_mask
        )

        vh_interior_zero = np.sum(
            vh_zero & interior_mask
        )

        vv_edge_zero_percentage = (
            vv_edge_zero
            / edge_pixels
            * 100
        )

        vh_edge_zero_percentage = (
            vh_edge_zero
            / edge_pixels
            * 100
        )

        vv_interior_zero_percentage = (
            vv_interior_zero
            / interior_pixels
            * 100
        )

        vh_interior_zero_percentage = (
            vh_interior_zero
            / interior_pixels
            * 100
        )

        # ----------------------------------------------------
        # Basic statistics excluding zero
        # ----------------------------------------------------

        vv_nonzero = vv[
            vv != 0
        ]

        vh_nonzero = vh[
            vh != 0
        ]

        return {
            "path": str(path),

            "nodata": nodata_values,

            "total_pixels": total_pixels,

            "vv_zero_count": vv_zero_count,
            "vh_zero_count": vh_zero_count,

            "vv_zero_percentage":
                vv_zero_percentage,

            "vh_zero_percentage":
                vh_zero_percentage,

            "vv_edge_zero_percentage":
                vv_edge_zero_percentage,

            "vh_edge_zero_percentage":
                vh_edge_zero_percentage,

            "vv_interior_zero_percentage":
                vv_interior_zero_percentage,

            "vh_interior_zero_percentage":
                vh_interior_zero_percentage,

            "vv_nonzero_min":
                float(np.min(vv_nonzero))
                if len(vv_nonzero) > 0
                else None,

            "vv_nonzero_max":
                float(np.max(vv_nonzero))
                if len(vv_nonzero) > 0
                else None,

            "vh_nonzero_min":
                float(np.min(vh_nonzero))
                if len(vh_nonzero) > 0
                else None,

            "vh_nonzero_max":
                float(np.max(vh_nonzero))
                if len(vh_nonzero) > 0
                else None,
        }


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(
    name,
    image_folder
):

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
    # Storage
    # --------------------------------------------------------

    vv_zero_percentages = []
    vh_zero_percentages = []

    vv_edge_zero_percentages = []
    vh_edge_zero_percentages = []

    vv_interior_zero_percentages = []
    vh_interior_zero_percentages = []

    nodata_values_found = set()

    images_with_vv_zero = 0
    images_with_vh_zero = 0

    images_with_many_vv_zero = 0
    images_with_many_vh_zero = 0

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for index, image_path in enumerate(images):

        try:

            result = analyze_image(
                image_path
            )

            # ------------------------------------------------
            # Zero percentages
            # ------------------------------------------------

            vv_zero_percentages.append(
                result[
                    "vv_zero_percentage"
                ]
            )

            vh_zero_percentages.append(
                result[
                    "vh_zero_percentage"
                ]
            )

            # ------------------------------------------------
            # Edge percentages
            # ------------------------------------------------

            vv_edge_zero_percentages.append(
                result[
                    "vv_edge_zero_percentage"
                ]
            )

            vh_edge_zero_percentages.append(
                result[
                    "vh_edge_zero_percentage"
                ]
            )

            # ------------------------------------------------
            # Interior percentages
            # ------------------------------------------------

            vv_interior_zero_percentages.append(
                result[
                    "vv_interior_zero_percentage"
                ]
            )

            vh_interior_zero_percentages.append(
                result[
                    "vh_interior_zero_percentage"
                ]
            )

            # ------------------------------------------------
            # NoData metadata
            # ------------------------------------------------

            for value in result["nodata"]:

                nodata_values_found.add(
                    value
                )

            # ------------------------------------------------
            # Presence of zeros
            # ------------------------------------------------

            if (
                result["vv_zero_count"]
                > 0
            ):

                images_with_vv_zero += 1

            if (
                result["vh_zero_count"]
                > 0
            ):

                images_with_vh_zero += 1

            # ------------------------------------------------
            # Images with >1% zero pixels
            # ------------------------------------------------

            if (
                result["vv_zero_percentage"]
                > 1
            ):

                images_with_many_vv_zero += 1

            if (
                result["vh_zero_percentage"]
                > 1
            ):

                images_with_many_vh_zero += 1

        except Exception as error:

            print(
                f"\nERROR: {image_path}"
            )

            print(error)

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
    # CONVERT TO ARRAYS
    # ========================================================

    vv_zero_percentages = np.asarray(
        vv_zero_percentages
    )

    vh_zero_percentages = np.asarray(
        vh_zero_percentages
    )

    vv_edge_zero_percentages = np.asarray(
        vv_edge_zero_percentages
    )

    vh_edge_zero_percentages = np.asarray(
        vh_edge_zero_percentages
    )

    vv_interior_zero_percentages = np.asarray(
        vv_interior_zero_percentages
    )

    vh_interior_zero_percentages = np.asarray(
        vh_interior_zero_percentages
    )

    # ========================================================
    # NODATA METADATA
    # ========================================================

    print("\n" + "-" * 75)
    print("NODATA METADATA")
    print("-" * 75)

    print(
        "NoData values found:"
    )

    for value in sorted(
        nodata_values_found,
        key=lambda x: str(x)
    ):

        print(
            f"  {value}"
        )

    # ========================================================
    # ZERO PIXEL PRESENCE
    # ========================================================

    print("\n" + "-" * 75)
    print("ZERO PIXEL PRESENCE")
    print("-" * 75)

    print(
        f"Images containing VV zeros: "
        f"{images_with_vv_zero}/"
        f"{len(images)} "
        f"("
        f"{images_with_vv_zero / len(images) * 100:.2f}%"
        f")"
    )

    print(
        f"Images containing VH zeros: "
        f"{images_with_vh_zero}/"
        f"{len(images)} "
        f"("
        f"{images_with_vh_zero / len(images) * 100:.2f}%"
        f")"
    )

    print(
        f"\nImages with >1% VV zeros: "
        f"{images_with_many_vv_zero}/"
        f"{len(images)}"
    )

    print(
        f"Images with >1% VH zeros: "
        f"{images_with_many_vh_zero}/"
        f"{len(images)}"
    )

    # ========================================================
    # ZERO PERCENTAGE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("ZERO PIXEL PERCENTAGE PER IMAGE")
    print("-" * 75)

    percentiles = [
        0,
        25,
        50,
        75,
        95,
        99,
        100
    ]

    print("\nVV:")

    for percentile in percentiles:

        value = np.percentile(
            vv_zero_percentages,
            percentile
        )

        print(
            f"  P{percentile}: "
            f"{value:.6f}%"
        )

    print("\nVH:")

    for percentile in percentiles:

        value = np.percentile(
            vh_zero_percentages,
            percentile
        )

        print(
            f"  P{percentile}: "
            f"{value:.6f}%"
        )

    # ========================================================
    # EDGE VS INTERIOR
    # ========================================================

    print("\n" + "-" * 75)
    print("EDGE VS INTERIOR ZERO DISTRIBUTION")
    print("-" * 75)

    print(
        f"Edge width: "
        f"{EDGE_WIDTH} pixels"
    )

    print("\nVV:")

    print(
        f"  Median edge zero: "
        f"{np.median(vv_edge_zero_percentages):.6f}%"
    )

    print(
        f"  Median interior zero: "
        f"{np.median(vv_interior_zero_percentages):.6f}%"
    )

    print(
        f"  P95 edge zero: "
        f"{np.percentile(vv_edge_zero_percentages, 95):.6f}%"
    )

    print(
        f"  P95 interior zero: "
        f"{np.percentile(vv_interior_zero_percentages, 95):.6f}%"
    )

    print("\nVH:")

    print(
        f"  Median edge zero: "
        f"{np.median(vh_edge_zero_percentages):.6f}%"
    )

    print(
        f"  Median interior zero: "
        f"{np.median(vh_interior_zero_percentages):.6f}%"
    )

    print(
        f"  P95 edge zero: "
        f"{np.percentile(vh_edge_zero_percentages, 95):.6f}%"
    )

    print(
        f"  P95 interior zero: "
        f"{np.percentile(vh_interior_zero_percentages, 95):.6f}%"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 75)
    print("DATASET ZERO SUMMARY")
    print("-" * 75)

    print(
        f"VV median zero percentage: "
        f"{np.median(vv_zero_percentages):.6f}%"
    )

    print(
        f"VH median zero percentage: "
        f"{np.median(vh_zero_percentages):.6f}%"
    )

    print(
        f"VV maximum zero percentage: "
        f"{np.max(vv_zero_percentages):.6f}%"
    )

    print(
        f"VH maximum zero percentage: "
        f"{np.max(vh_zero_percentages):.6f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "             ZERO / NODATA VALIDATION"
    )
    print("=" * 75)

    config = load_config()

    dataset = config["dataset"]

    # ========================================================
    # PART 1 - OIL
    # ========================================================

    analyze_dataset(
        "PART 1 - OIL",
        dataset["oil"]["images"]
    )

    # ========================================================
    # PART 2 - NO OIL
    # ========================================================

    analyze_dataset(
        "PART 2 - NO OIL",
        dataset["no_oil"]["images"]
    )

    # ========================================================
    # PART 2 - LOOKALIKE
    # ========================================================

    analyze_dataset(
        "PART 2 - LOOKALIKE",
        dataset["lookalike"]["images"]
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "ZERO / NODATA VALIDATION COMPLETE"
    )
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()