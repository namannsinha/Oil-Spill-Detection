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
# CONFIG LOADING
# ============================================================

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# TIFF DISCOVERY
# ============================================================

def find_tiffs(folder):
    folder = Path(folder)

    return sorted(
        list(folder.rglob("*.tif")) +
        list(folder.rglob("*.tiff"))
    )


# ============================================================
# IMAGE INSPECTION
# ============================================================

def inspect_image(path):

    print(f"\n{'-' * 60}")
    print(f"IMAGE: {path}")

    try:

        with rasterio.open(path) as src:

            print(f"Shape       : ({src.count}, {src.height}, {src.width})")
            print(f"Bands       : {src.count}")
            print(f"Data types  : {src.dtypes}")
            print(f"CRS         : {src.crs}")
            print(f"Georeference: {'YES' if src.transform else 'NO'}")

            for band in range(1, src.count + 1):

                data = src.read(band)

                finite = data[np.isfinite(data)]

                print(f"\nBand {band}")
                print(f"  Min  : {finite.min():.4f}")
                print(f"  Max  : {finite.max():.4f}")
                print(f"  Mean : {finite.mean():.4f}")
                print(f"  Std  : {finite.std():.4f}")
                print(f"  NaN  : {np.isnan(data).sum()}")

    except Exception as e:

        print(f"ERROR: {e}")


# ============================================================
# MASK INSPECTION
# ============================================================

def inspect_mask(path):

    print(f"\n{'-' * 60}")
    print(f"MASK: {path}")

    try:

        with rasterio.open(path) as src:

            data = src.read(1)

            unique, counts = np.unique(
                data,
                return_counts=True
            )

            print(f"Shape      : {data.shape}")
            print(f"Data type  : {data.dtype}")
            print(f"CRS        : {src.crs}")

            print("\nPixel values:")

            for value, count in zip(unique, counts):

                percentage = count / data.size * 100

                print(
                    f"  {value}: "
                    f"{count:,} pixels "
                    f"({percentage:.4f}%)"
                )

    except Exception as e:

        print(f"ERROR: {e}")


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(name, images_path, masks_path):

    images = find_tiffs(images_path)
    masks = find_tiffs(masks_path)

    print("\n")
    print("=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    print(f"Images: {len(images)}")
    print(f"Masks : {len(masks)}")

    # --------------------------------------------------------
    # Filename matching
    # --------------------------------------------------------

    image_names = {file.name for file in images}
    mask_names = {file.name for file in masks}

    missing_masks = image_names - mask_names
    missing_images = mask_names - image_names

    print(f"\nMissing masks : {len(missing_masks)}")
    print(f"Missing images: {len(missing_images)}")

    if missing_masks:

        print("\nImages without masks:")

        for file in sorted(missing_masks)[:20]:
            print(f"  {file}")

    if missing_images:

        print("\nMasks without images:")

        for file in sorted(missing_images)[:20]:
            print(f"  {file}")

    # --------------------------------------------------------
    # Inspect first image
    # --------------------------------------------------------

    if images:

        print("\n\nFIRST IMAGE")

        inspect_image(images[0])

    # --------------------------------------------------------
    # Inspect first mask
    # --------------------------------------------------------

    if masks:

        print("\n\nFIRST MASK")

        inspect_mask(masks[0])

    return {
        "images": len(images),
        "masks": len(masks),
        "missing_masks": len(missing_masks),
        "missing_images": len(missing_images),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("        OIL SPILL DETECTION")
    print("        DATASET INSPECTION")
    print("=" * 70)

    config = load_config()

    dataset_config = config["dataset"]

    results = {}

    # --------------------------------------------------------
    # Oil
    # --------------------------------------------------------

    results["oil"] = validate_dataset(
        "PART 1 - OIL",
        dataset_config["oil"]["images"],
        dataset_config["oil"]["masks"],
    )

    # --------------------------------------------------------
    # No Oil
    # --------------------------------------------------------

    results["no_oil"] = validate_dataset(
        "PART 2 - NO OIL",
        dataset_config["no_oil"]["images"],
        dataset_config["no_oil"]["masks"],
    )

    # --------------------------------------------------------
    # Lookalike
    # --------------------------------------------------------

    results["lookalike"] = validate_dataset(
        "PART 2 - LOOKALIKE",
        dataset_config["lookalike"]["images"],
        dataset_config["lookalike"]["masks"],
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for name, result in results.items():

        status = (
            "PASS"
            if (
                result["images"] == result["masks"]
                and result["missing_masks"] == 0
                and result["missing_images"] == 0
            )
            else "CHECK"
        )

        print(
            f"{name:12} | "
            f"Images: {result['images']:4} | "
            f"Masks: {result['masks']:4} | "
            f"{status}"
        )


if __name__ == "__main__":
    main()