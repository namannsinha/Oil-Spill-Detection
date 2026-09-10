from pathlib import Path
import rasterio
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# ============================================================
# HELPERS
# ============================================================

def find_tiffs(folder):
    """Recursively find all TIFF files."""
    return sorted(
        list(folder.rglob("*.tif")) +
        list(folder.rglob("*.tiff"))
    )


def inspect_tiff(path):
    """Read basic information from a TIFF without loading
    the entire dataset into memory unnecessarily.
    """

    try:
        with rasterio.open(path) as src:

            print(f"\nFile: {path}")
            print(f"  Shape       : ({src.count}, {src.height}, {src.width})")
            print(f"  Bands       : {src.count}")
            print(f"  Width       : {src.width}")
            print(f"  Height      : {src.height}")
            print(f"  Dtype       : {src.dtypes}")
            print(f"  CRS         : {src.crs}")
            print(f"  Transform   : {src.transform}")

            for band_number in range(1, src.count + 1):

                data = src.read(band_number)

                finite_data = data[np.isfinite(data)]

                print(
                    f"  Band {band_number}: "
                    f"min={finite_data.min():.4f}, "
                    f"max={finite_data.max():.4f}, "
                    f"mean={finite_data.mean():.4f}, "
                    f"std={finite_data.std():.4f}"
                )

            return True

    except Exception as e:

        print(f"\nERROR reading: {path}")
        print(f"       {e}")

        return False


def inspect_mask(path):
    """Inspect mask values."""

    try:
        with rasterio.open(path) as src:

            data = src.read(1)

            unique, counts = np.unique(
                data,
                return_counts=True
            )

            print(f"\nMask: {path}")
            print(f"  Shape : ({src.height}, {src.width})")
            print(f"  Dtype : {src.dtypes[0]}")

            print("  Values:")

            for value, count in zip(unique, counts):

                percentage = (
                    count / data.size
                ) * 100

                print(
                    f"    {value}: "
                    f"{count:,} pixels "
                    f"({percentage:.4f}%)"
                )

            return True

    except Exception as e:

        print(f"\nERROR reading mask: {path}")
        print(f"       {e}")

        return False


# ============================================================
# DATASET DISCOVERY
# ============================================================

def show_directory_structure():

    print("\n" + "=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)

    if not DATA_ROOT.exists():

        print(f"ERROR: {DATA_ROOT} does not exist.")
        return

    for path in sorted(DATA_ROOT.rglob("*")):

        if path.is_file():

            relative = path.relative_to(DATA_ROOT)

            print(f"  {relative}")


# ============================================================
# DATASET SUMMARY
# ============================================================

def summarize_dataset():

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    part1 = DATA_ROOT / "part1"
    part2 = DATA_ROOT / "part2"

    datasets = {
        "PART 1": part1,
        "PART 2": part2,
    }

    for name, folder in datasets.items():

        print(f"\n{name}")
        print("-" * 50)

        if not folder.exists():

            print(f"Folder not found: {folder}")
            continue

        files = find_tiffs(folder)

        print(f"Total TIFF files: {len(files)}")

        # Group files according to their parent folder
        groups = {}

        for file in files:

            parent = file.parent.relative_to(folder)

            groups.setdefault(parent, 0)
            groups[parent] += 1

        for group, count in groups.items():

            print(f"  {group}: {count} TIFFs")


# ============================================================
# SAMPLE INSPECTION
# ============================================================

def inspect_samples():

    print("\n" + "=" * 70)
    print("SAMPLE FILE INSPECTION")
    print("=" * 70)

    files = find_tiffs(DATA_ROOT)

    if not files:

        print("\nNo TIFF files found.")
        print(f"Expected them somewhere inside: {DATA_ROOT}")
        return

    # Inspect first few files only
    sample_files = files[:5]

    for file in sample_files:

        inspect_tiff(file)


# ============================================================
# MASK INSPECTION
# ============================================================

def inspect_mask_samples():

    print("\n" + "=" * 70)
    print("MASK SAMPLE INSPECTION")
    print("=" * 70)

    files = find_tiffs(DATA_ROOT)

    mask_files = [
        file for file in files
        if "mask" in str(file).lower()
    ]

    if not mask_files:

        print("\nNo files containing 'mask' in their path were found.")

        return

    for file in mask_files[:5]:

        inspect_mask(file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("        OIL SPILL DETECTION - DATASET INSPECTOR")
    print("=" * 70)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Data root    : {DATA_ROOT}")

    show_directory_structure()

    summarize_dataset()

    inspect_samples()

    inspect_mask_samples()

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()