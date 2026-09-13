from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVENT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "environmental"
    / "event_2017_10_17"
)

OUTPUT_DIR = EVENT_DIR / "ais"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Event
# ============================================================

SPILL_LAT = 34.77504313874554
SPILL_LON = 30.372601197819165

# Analyze a 25 km radius around the T0 spill centroid.
RADIUS_KM = 25

YEAR = 2017
MONTH = 10


# ============================================================
# EMODnet
# ============================================================

# We will download the relevant raster through EMODnet WCS.
WCS_URL = "https://ows.emodnet-humanactivities.eu/wcs"

COVERAGE_ID = "emodnet__vesseldensity_all"


# ============================================================
# Coordinate conversion
# ============================================================

# EMODnet vessel-density raster is published in a projected
# coordinate system. We request a geographic bounding box and
# let the WCS server return the raster.


def calculate_bbox(lat, lon, radius_km):
    """
    Approximate geographic bounding box around a point.
    """

    lat_delta = radius_km / 111.32

    lon_delta = radius_km / (
        111.32 * np.cos(np.radians(lat))
    )

    return (
        lon - lon_delta,
        lat - lat_delta,
        lon + lon_delta,
        lat + lat_delta,
    )


# ============================================================
# Download
# ============================================================

def download_vessel_density():
    """
    Download October 2017 EMODnet vessel-density raster.

    Uses the EMODnet WCS service.
    """

    import requests

    west, south, east, north = calculate_bbox(
        SPILL_LAT,
        SPILL_LON,
        RADIUS_KM,
    )

    output_file = OUTPUT_DIR / "vessel_density_2017_10.tif"

    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": COVERAGE_ID,

        # Request geographic output.
        "subset": [
            f"Long({west},{east})",
            f"Lat({south},{north})",
            f"time(2017-10-01T00:00:00Z,2017-10-01T00:00:00Z)",
        ],

        "format": "image/tiff",
    }

    print("\nDownloading EMODnet vessel density...")
    print(f"Coverage: {COVERAGE_ID}")
    print(
        f"Bounding box: "
        f"{west:.4f}, {south:.4f}, "
        f"{east:.4f}, {north:.4f}"
    )

    response = requests.get(
        WCS_URL,
        params=params,
        timeout=120,
    )

    print("HTTP status:", response.status_code)

    response.raise_for_status()

    output_file.write_bytes(response.content)

    print("Saved:")
    print(output_file)

    return output_file


# ============================================================
# Analysis
# ============================================================

def analyze_density(path):
    """
    Calculate maritime activity statistics around spill.
    """

    print("\nAnalyzing vessel density...")

    with rasterio.open(path) as src:

        data = src.read(1).astype(np.float64)

        nodata = src.nodata

        if nodata is not None:
            data[data == nodata] = np.nan

        data[data < 0] = np.nan

        valid = np.isfinite(data)

        if not np.any(valid):
            raise ValueError(
                "No valid vessel-density data found."
            )

        result = {
            "source": "EMODnet Human Activities",
            "coverage": COVERAGE_ID,
            "year": YEAR,
            "month": MONTH,
            "spill_latitude": SPILL_LAT,
            "spill_longitude": SPILL_LON,
            "radius_km": RADIUS_KM,

            "valid_cells": int(valid.sum()),

            "mean_density": float(
                np.nanmean(data)
            ),

            "max_density": float(
                np.nanmax(data)
            ),

            "total_density": float(
                np.nansum(data)
            ),
        }

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("AIS / VESSEL ACTIVITY ANALYSIS")
    print("=" * 70)

    raster = download_vessel_density()

    result = analyze_density(raster)

    output_json = OUTPUT_DIR / "ais_analysis.json"

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\nAIS ANALYSIS")
    print("=" * 70)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(output_json)


if __name__ == "__main__":
    main()