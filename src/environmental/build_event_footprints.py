from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Geod


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data/events/part1_event_manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs/environmental/events"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T0_PRODUCT = (
    "S1A_IW_GRDH_1SDV_20171017T155749_"
    "20171017T155814_018855_01FD8C_9271_O"
)

T1_PRODUCT = (
    "S1A_IW_GRDH_1SDV_20171018T035948_"
    "20171018T040013_018862_01FDC4_C249_O"
)

GEOD = Geod(ellps="WGS84")


# ============================================================
# HELPERS
# ============================================================

def geodesic_area_km2(geometry):
    """
    Calculate geodesic area of a WGS84 geometry.
    """
    if geometry is None or geometry.is_empty:
        return 0.0

    area, _ = GEOD.geometry_area_perimeter(geometry)

    return abs(area) / 1_000_000.0


def raster_mask_to_geometry(mask_path, image_path):
    """
    Convert a binary oil mask into a georeferenced polygon
    using the corresponding Sentinel-1 image transform.

    IMPORTANT:
    The mask itself may not contain valid georeferencing.
    Therefore we deliberately use the IMAGE transform.
    """

    with rasterio.open(image_path) as image_src:
        transform = image_src.transform
        crs = image_src.crs

    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)

    # Ensure binary mask
    binary_mask = mask > 0

    if not binary_mask.any():
        return None, crs

    geometries = []

    for geom, value in shapes(
        binary_mask.astype(np.uint8),
        mask=binary_mask,
        transform=transform,
    ):
        if value == 1:
            geometries.append(shape(geom))

    if not geometries:
        return None, crs

    merged = unary_union(geometries)

    return merged, crs


def build_product_footprint(product_id, records):
    """
    Merge all oil-mask geometries belonging to one Sentinel-1
    product.
    """

    geometries = []
    scene_ids = []

    for _, row in records.iterrows():

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        if not image_path.exists():
            print(f"[WARNING] Image missing: {image_path}")
            continue

        if not mask_path.exists():
            print(f"[WARNING] Mask missing: {mask_path}")
            continue

        geometry, crs = raster_mask_to_geometry(
            mask_path,
            image_path,
        )

        if geometry is not None and not geometry.is_empty:
            geometries.append(geometry)
            scene_ids.append(int(row["scene_id"]))

    if not geometries:
        raise RuntimeError(
            f"No valid oil geometry found for product {product_id}"
        )

    # Merge all scene geometries
    footprint = unary_union(geometries)

    area_km2 = geodesic_area_km2(footprint)

    centroid = footprint.centroid

    bounds = footprint.bounds

    return {
        "product_id": product_id,
        "scene_count": len(scene_ids),
        "scene_ids": scene_ids,
        "geometry": footprint,
        "area_km2": area_km2,
        "centroid_latitude": centroid.y,
        "centroid_longitude": centroid.x,
        "min_longitude": bounds[0],
        "min_latitude": bounds[1],
        "max_longitude": bounds[2],
        "max_latitude": bounds[3],
        "crs": str(crs),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING HISTORICAL OIL-SPILL EVENT FOOTPRINTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = pd.read_csv(MANIFEST_PATH)

    print(f"\nManifest rows: {len(df)}")

    # --------------------------------------------------------
    # Extract T0/T1 records
    # --------------------------------------------------------

    t0_records = df[df["product_id"] == T0_PRODUCT].copy()
    t1_records = df[df["product_id"] == T1_PRODUCT].copy()

    print(f"T0 scenes: {len(t0_records)}")
    print(f"T1 scenes: {len(t1_records)}")

    if t0_records.empty:
        raise RuntimeError("T0 product not found in manifest.")

    if t1_records.empty:
        raise RuntimeError("T1 product not found in manifest.")

    # --------------------------------------------------------
    # Build footprints
    # --------------------------------------------------------

    print("\nBuilding T0 footprint...")
    t0 = build_product_footprint(
        T0_PRODUCT,
        t0_records,
    )

    print("Building T1 footprint...")
    t1 = build_product_footprint(
        T1_PRODUCT,
        t1_records,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("T0 FOOTPRINT")
    print("=" * 70)

    print(f"Scenes       : {t0['scene_count']}")
    print(f"Area         : {t0['area_km2']:.4f} km²")
    print(
        f"Centroid     : "
        f"{t0['centroid_latitude']:.6f}, "
        f"{t0['centroid_longitude']:.6f}"
    )

    print("\n" + "=" * 70)
    print("T1 FOOTPRINT")
    print("=" * 70)

    print(f"Scenes       : {t1['scene_count']}")
    print(f"Area         : {t1['area_km2']:.4f} km²")
    print(
        f"Centroid     : "
        f"{t1['centroid_latitude']:.6f}, "
        f"{t1['centroid_longitude']:.6f}"
    )

    # --------------------------------------------------------
    # Calculate centroid displacement
    # --------------------------------------------------------

    _, _, distance_m = GEOD.inv(
        t0["centroid_longitude"],
        t0["centroid_latitude"],
        t1["centroid_longitude"],
        t1["centroid_latitude"],
    )

    displacement_km = distance_m / 1000.0

    print("\n" + "=" * 70)
    print("OBSERVED MOVEMENT")
    print("=" * 70)

    print(f"Centroid displacement: {displacement_km:.4f} km")

    # --------------------------------------------------------
    # Create GeoJSON
    # --------------------------------------------------------

    def save_geojson(result, filename):

        feature = {
            "type": "Feature",
            "properties": {
                "product_id": result["product_id"],
                "scene_count": result["scene_count"],
                "scene_ids": result["scene_ids"],
                "area_km2": result["area_km2"],
                "centroid_latitude": result["centroid_latitude"],
                "centroid_longitude": result["centroid_longitude"],
                "crs": result["crs"],
            },
            "geometry": mapping(result["geometry"]),
        }

        geojson = {
            "type": "FeatureCollection",
            "features": [feature],
        }

        path = OUTPUT_DIR / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        print(f"\nSaved: {path}")

    save_geojson(
        t0,
        "t0_footprint.geojson",
    )

    save_geojson(
        t1,
        "t1_footprint.geojson",
    )

    # --------------------------------------------------------
    # Save event pair
    # --------------------------------------------------------

    t0_time = pd.to_datetime(
        t0_records["timestamp_start"]
    ).min()

    t1_time = pd.to_datetime(
        t1_records["timestamp_start"]
    ).min()

    time_delta_hours = (
        (t1_time - t0_time).total_seconds()
        / 3600.0
    )

    event_pair = {
        "event_id": "20171017_20171018_001",

        "t0": {
            "product_id": T0_PRODUCT,
            "timestamp": t0_time.isoformat(),
            "scene_count": t0["scene_count"],
            "area_km2": t0["area_km2"],
            "centroid": {
                "latitude": t0["centroid_latitude"],
                "longitude": t0["centroid_longitude"],
            },
            "geojson": "t0_footprint.geojson",
        },

        "t1": {
            "product_id": T1_PRODUCT,
            "timestamp": t1_time.isoformat(),
            "scene_count": t1["scene_count"],
            "area_km2": t1["area_km2"],
            "centroid": {
                "latitude": t1["centroid_latitude"],
                "longitude": t1["centroid_longitude"],
            },
            "geojson": "t1_footprint.geojson",
        },

        "time_delta_hours": time_delta_hours,

        "observed_centroid_displacement_km": displacement_km,
    }

    pair_path = OUTPUT_DIR / "event_pair.json"

    with open(pair_path, "w", encoding="utf-8") as f:
        json.dump(event_pair, f, indent=2)

    print(f"Saved: {pair_path}")

    print("\n" + "=" * 70)
    print("FOOTPRINT BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()