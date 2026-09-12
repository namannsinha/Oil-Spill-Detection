"""
Discover Part I oil-spill events.

Extracts:
- Sentinel-1 acquisition timestamps from custom TIFF tag 65000
- Sentinel-1 product name
- Scene geolocation
- Ground-truth oil statistics
- Oil centroid
- Estimated oil area
- Event suitability score

Output:
    data/events/part1_event_manifest.csv
"""

from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import tifffile


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PART1_ROOT = Path(
    r"D:\SIH_Dataset\Part_I"
)

IMAGE_ROOT = (
    PART1_ROOT
    / "01_Train_Val_Oil_Spill_images"
    / "Oil"
)

MASK_ROOT = (
    PART1_ROOT
    / "01_Train_Val_Oil_Spill_mask"
    / "Mask_oil"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "events"
OUTPUT_CSV = OUTPUT_DIR / "part1_event_manifest.csv"


# ============================================================
# HELPERS
# ============================================================


def find_mask(image_path: Path) -> Path:
    """
    Find the ground-truth mask corresponding to an image.
    """

    mask_path = MASK_ROOT / image_path.name

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found for {image_path.name}: {mask_path}"
        )

    return mask_path


def parse_dimap_metadata(image_path: Path) -> dict:
    """
    Read custom TIFF tag 65000 containing DIMAP XML.
    """

    with tifffile.TiffFile(image_path) as tif:
        page = tif.pages[0]

        if 65000 not in page.tags:
            raise ValueError(
                f"TIFF tag 65000 not found: {image_path}"
            )

        xml_text = page.tags[65000].value

    root = ET.fromstring(xml_text)

    metadata = {}

    for element in root.iter():
        if element.text:
            metadata[element.tag] = element.text.strip()

    return metadata


def parse_datetime(value: str | None) -> str | None:
    """
    Convert DIMAP date string into ISO format.
    """

    if not value:
        return None

    formats = [
        "%d-%b-%Y %H:%M:%S.%f",
        "%d-%b-%Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(
                value.upper(),
                fmt
            )
            return dt.isoformat()
        except ValueError:
            pass

    return value


def extract_product_id(dataset_name: str | None) -> str | None:
    """
    Extract the Sentinel-1 product identifier from DATASET_NAME.

    Example:
    subset_0_of_S1A_IW_GRDH_1SDV_20180803T172551_...
    ->
    S1A_IW_GRDH_1SDV_20180803T172551_...
    """

    if not dataset_name:
        return None

    match = re.search(
        r"(S1[AB]_[A-Z0-9_]+)",
        dataset_name
    )

    if match:
        return match.group(1)

    return dataset_name


def scene_geolocation(image_path: Path) -> dict:
    """
    Extract scene bounds and centroid.
    """

    with rasterio.open(image_path) as src:

        bounds = src.bounds

        scene_latitude = (
            bounds.top + bounds.bottom
        ) / 2.0

        scene_longitude = (
            bounds.left + bounds.right
        ) / 2.0

        return {
            "crs": str(src.crs) if src.crs else None,
            "width": src.width,
            "height": src.height,
            "scene_left": bounds.left,
            "scene_bottom": bounds.bottom,
            "scene_right": bounds.right,
            "scene_top": bounds.top,
            "scene_latitude": scene_latitude,
            "scene_longitude": scene_longitude,
        }


def calculate_oil_statistics(mask_path: Path) -> dict:
    """
    Calculate oil pixel statistics and oil centroid.

    Mask convention:
        0 = background
        >0 = oil
    """

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

        transform = src.transform

    oil = mask > 0

    total_pixels = oil.size
    oil_pixels = int(oil.sum())

    oil_percent = (
        oil_pixels / total_pixels * 100.0
        if total_pixels
        else 0.0
    )

    if oil_pixels == 0:
        return {
            "oil_pixels": 0,
            "oil_percent": 0.0,
            "oil_latitude": None,
            "oil_longitude": None,
            "oil_area_km2": 0.0,
        }

    rows, cols = np.where(oil)

    # Pixel center coordinates.
    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center",
    )

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    oil_longitude = float(xs.mean())
    oil_latitude = float(ys.mean())

    # --------------------------------------------------------
    # Estimate geographic pixel area.
    #
    # Dataset is EPSG:4326, therefore pixel dimensions are in
    # degrees. Approximate each row's pixel area on WGS84.
    # --------------------------------------------------------

    pixel_width_deg = abs(transform.a)
    pixel_height_deg = abs(transform.e)

    latitudes = np.asarray(
        rasterio.transform.xy(
            transform,
            rows,
            np.zeros_like(rows),
            offset="center",
        )[1],
        dtype=np.float64,
    )

    # WGS84 approximation.
    lat_rad = np.radians(latitudes)

    earth_radius = 6378137.0

    # Longitude degree length.
    meters_per_degree_lon = (
        np.pi
        * earth_radius
        * np.cos(lat_rad)
        / 180.0
    )

    # Latitude degree length.
    meters_per_degree_lat = (
        np.pi
        * earth_radius
        / 180.0
    )

    pixel_areas_m2 = (
        pixel_width_deg
        * meters_per_degree_lon
        * pixel_height_deg
        * meters_per_degree_lat
    )

    oil_area_km2 = (
        float(pixel_areas_m2.sum())
        / 1_000_000.0
    )

    return {
        "oil_pixels": oil_pixels,
        "oil_percent": oil_percent,
        "oil_latitude": oil_latitude,
        "oil_longitude": oil_longitude,
        "oil_area_km2": oil_area_km2,
    }


def calculate_event_score(
    oil_percent: float,
    oil_area_km2: float,
) -> float:
    """
    Preliminary event suitability score.

    This is NOT a scientific model score.

    It simply favors:
    - non-trivial spills
    - reasonably sized spills

    Final event ranking will additionally consider:
    - temporal sequences
    - environmental coverage
    - AIS availability
    - geographic suitability
    """

    if oil_percent <= 0:
        return 0.0

    # Prefer scenes with 1-10% oil coverage.
    if oil_percent < 0.25:
        size_score = 0.2
    elif oil_percent < 1:
        size_score = 0.5
    elif oil_percent < 10:
        size_score = 1.0
    elif oil_percent < 25:
        size_score = 0.9
    else:
        size_score = 0.7

    # Avoid selecting extremely tiny or enormous events only.
    if oil_area_km2 < 0.1:
        area_score = 0.2
    elif oil_area_km2 < 1:
        area_score = 0.5
    elif oil_area_km2 < 100:
        area_score = 1.0
    else:
        area_score = 0.8

    return round(
        100.0 * size_score * area_score,
        4,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PART I EVENT DISCOVERY")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Part I root  : {PART1_ROOT}")
    print(f"Image root   : {IMAGE_ROOT}")
    print()

    image_files = sorted(
        IMAGE_ROOT.glob("*.tif")
    )

    print(
        f"Found {len(image_files)} Part I oil images."
    )
    mask_files = sorted(
    MASK_ROOT.glob("*.tif")
)

    print(
        f"Found {len(mask_files)} Part I oil masks."
    )

    if len(mask_files) != len(image_files):
        raise RuntimeError(
            f"Image/mask count mismatch: "
            f"{len(image_files)} images vs "
            f"{len(mask_files)} masks"
        )

    print(f"Mask root    : {MASK_ROOT}")
    print()

    if len(image_files) == 0:
        raise RuntimeError(
            "No Part I images found."
        )


    rows = []

    failures = []

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        scene_id = image_path.stem

        try:

            mask_path = find_mask(image_path)

            metadata = parse_dimap_metadata(
                image_path
            )

            geo = scene_geolocation(
                image_path
            )

            oil = calculate_oil_statistics(
                mask_path
            )

            dataset_name = metadata.get(
                "DATASET_NAME"
            )

            timestamp_start = parse_datetime(
                metadata.get(
                    "PRODUCT_SCENE_RASTER_START_TIME"
                )
            )

            timestamp_stop = parse_datetime(
                metadata.get(
                    "PRODUCT_SCENE_RASTER_STOP_TIME"
                )
            )

            product_id = extract_product_id(
                dataset_name
            )

            score = calculate_event_score(
                oil["oil_percent"],
                oil["oil_area_km2"],
            )

            row = {
                "scene_id": scene_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),

                "product_id": product_id,
                "dataset_name": dataset_name,
                "product_type": metadata.get(
                    "PRODUCT_TYPE"
                ),

                "timestamp_start": timestamp_start,
                "timestamp_stop": timestamp_stop,

                "crs": geo["crs"],
                "width": geo["width"],
                "height": geo["height"],

                "scene_left": geo["scene_left"],
                "scene_bottom": geo["scene_bottom"],
                "scene_right": geo["scene_right"],
                "scene_top": geo["scene_top"],

                "scene_latitude": geo[
                    "scene_latitude"
                ],
                "scene_longitude": geo[
                    "scene_longitude"
                ],

                "oil_pixels": oil[
                    "oil_pixels"
                ],
                "oil_percent": oil[
                    "oil_percent"
                ],
                "oil_area_km2": oil[
                    "oil_area_km2"
                ],

                "oil_latitude": oil[
                    "oil_latitude"
                ],
                "oil_longitude": oil[
                    "oil_longitude"
                ],

                "event_score": score,

                # To be populated later.
                "environment_available": None,
                "ais_available": None,
                "followup_scene_available": None,
            }

            rows.append(row)

            if index % 50 == 0 or index == len(image_files):
                print(
                    f"[{index:4d}/{len(image_files)}] "
                    f"processed | "
                    f"oil={oil['oil_percent']:.3f}% | "
                    f"area={oil['oil_area_km2']:.2f} km²"
                )

        except Exception as exc:

            failures.append({
                "scene_id": scene_id,
                "image_path": str(image_path),
                "error": str(exc),
            })

            print(
                f"[ERROR] {scene_id}: {exc}"
            )

    if not rows:
        raise RuntimeError(
            "No scenes were successfully processed."
        )

    # Sort by preliminary event score.
    rows.sort(
        key=lambda x: x["event_score"],
        reverse=True,
    )

    fieldnames = list(rows[0].keys())

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(f"Successful scenes : {len(rows)}")
    print(f"Failed scenes     : {len(failures)}")
    print(f"Output            : {OUTPUT_CSV}")

    if failures:

        failure_csv = (
            OUTPUT_DIR
            / "part1_event_discovery_failures.csv"
        )

        with failure_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scene_id",
                    "image_path",
                    "error",
                ],
            )

            writer.writeheader()
            writer.writerows(failures)

        print(
            f"Failures saved   : {failure_csv}"
        )

    print()
    print("TOP 20 PRELIMINARY EVENTS")
    print("-" * 70)

    for rank, row in enumerate(
        rows[:20],
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"{row['scene_id']} | "
            f"{row['timestamp_start']} | "
            f"oil={row['oil_percent']:.2f}% | "
            f"area={row['oil_area_km2']:.2f} km² | "
            f"score={row['event_score']:.1f}"
        )


if __name__ == "__main__":
    main()