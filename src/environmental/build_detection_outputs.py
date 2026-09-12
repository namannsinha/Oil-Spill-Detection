from pathlib import Path
import csv
import json

import rasterio
from pyproj import Geod

from src.environmental.detection_outputs import extract_spill_polygon


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "predictions"
PROBABILITIES_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "probabilities"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "environmental" / "detections"
GEOJSON_ROOT = OUTPUT_ROOT / "geojson"

CATEGORIES = ["Oil", "No_oil", "Lookalike"]

THRESHOLD = 0.5
MIN_PIXELS = 20

# WGS84 ellipsoid for accurate area calculation from lon/lat geometry.
GEOD = Geod(ellps="WGS84")


def geodesic_area_km2(geometry):
    """
    Calculate polygon/multipolygon area in km² using WGS84 geodesics.
    """
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]

    elif geometry["type"] == "MultiPolygon":
        polygons = [
            polygon
            for polygon in geometry["coordinates"]
        ]

    else:
        return 0.0

    total_area_m2 = 0.0

    for polygon in polygons:
        # polygon[0] = exterior ring
        exterior = polygon[0]

        lons = [point[0] for point in exterior]
        lats = [point[1] for point in exterior]

        area, _ = GEOD.polygon_area_perimeter(lons, lats)
        total_area_m2 += abs(area)

        # Subtract holes.
        for hole in polygon[1:]:
            lons = [point[0] for point in hole]
            lats = [point[1] for point in hole]

            hole_area, _ = GEOD.polygon_area_perimeter(lons, lats)
            total_area_m2 -= abs(hole_area)

    return total_area_m2 / 1_000_000.0


def get_timestamp(image_path):
    """
    Try to retrieve acquisition timestamp from available TIFF metadata.

    The current TIFFs may not contain acquisition time. In that case
    return None rather than inventing a timestamp.
    """
    timestamp_keys = [
        "datetime",
        "DateTime",
        "DATETIME",
        "acquisition_time",
        "acquisition_datetime",
        "sensing_time",
        "start_time",
        "TIFFTAG_DATETIME",
    ]

    with rasterio.open(image_path) as src:
        tags = src.tags()

        for key in timestamp_keys:
            if key in tags and tags[key]:
                return tags[key]

    return None


def process_scene(category, prediction_path):
    scene_id = prediction_path.stem

    probability_path = (
        PROBABILITIES_ROOT
        / category
        / f"{scene_id}.tif"
    )

    # Some datasets use No_oil in outputs.
    if not probability_path.exists():
        probability_path = (
            PROBABILITIES_ROOT
            / category.replace("No_oil", "No_oil")
            / f"{scene_id}.tif"
        )

    image_category = category.replace("No_oil", "No oil")

    source_image = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "part3"
        / "Images"
        / image_category
        / f"{scene_id}.tif"
    )

    # The Part III images actually live outside the project in the
    # current setup, so search the known dataset location if necessary.
    if not source_image.exists():
        source_image = (
            Path(r"D:\SIH_Dataset")
            / "Part_III"
            / "02_Test_images_and_ground_truth"
            / "Images"
            / image_category
            / f"{scene_id}.tif"
        )

    result = extract_spill_polygon(
        mask_path=prediction_path,
        probability_path=probability_path if probability_path.exists() else None,
        threshold=THRESHOLD,
        min_pixels=MIN_PIXELS,
    )

    geometry = result["geometry"]

    if geometry:
        area_km2 = geodesic_area_km2(geometry)

        centroid_lat = result["centroid"]["latitude"]
        centroid_lon = result["centroid"]["longitude"]
    else:
        area_km2 = 0.0
        centroid_lat = None
        centroid_lon = None

    timestamp = None

    if source_image.exists():
        timestamp = get_timestamp(source_image)

    output = {
        "scene_id": scene_id,
        "category": category,
        "timestamp": timestamp,
        "centroid": {
            "latitude": centroid_lat,
            "longitude": centroid_lon,
        },
        "area_km2": area_km2,
        "area_pixels": result["area_pixels"],
        "predicted_oil_percentage": result["predicted_oil_percentage"],
        "mean_confidence": result["mean_confidence"],
        "max_confidence": result["max_confidence"],
        "crs": str(result.get("crs")) if result.get("crs") else None,        
        "geometry": geometry,
    }

    return output


def save_geojson(result, output_path):
    """
    Save one spill detection as a GeoJSON FeatureCollection.
    """

    geometry = result["geometry"]

    properties = {
        "scene_id": result["scene_id"],
        "category": result["category"],
        "timestamp": result["timestamp"],
        "area_km2": result["area_km2"],
        "area_pixels": result["area_pixels"],
        "predicted_oil_percentage": result["predicted_oil_percentage"],
        "mean_confidence": result["mean_confidence"],
        "max_confidence": result["max_confidence"],
        "centroid_latitude": result["centroid"]["latitude"],
        "centroid_longitude": result["centroid"]["longitude"],
    }

    feature = {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties,
    }

    feature_collection = {
        "type": "FeatureCollection",
        "features": [feature],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)


def main():
    GEOJSON_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []

    for category in CATEGORIES:

        prediction_dir = PREDICTIONS_ROOT / category

        if not prediction_dir.exists():
            print(f"[WARNING] Missing: {prediction_dir}")
            continue

        output_dir = GEOJSON_ROOT / category
        output_dir.mkdir(parents=True, exist_ok=True)

        prediction_files = sorted(
            prediction_dir.glob("*.tif")
        )

        print(f"\n{category}: {len(prediction_files)} scenes")

        for index, prediction_path in enumerate(prediction_files, start=1):

            result = process_scene(
                category,
                prediction_path,
            )

            output_path = (
                output_dir
                / f"{result['scene_id']}.geojson"
            )

            save_geojson(
                result,
                output_path,
            )

            rows.append({
                "scene_id": result["scene_id"],
                "category": result["category"],
                "timestamp": result["timestamp"],
                "latitude": result["centroid"]["latitude"],
                "longitude": result["centroid"]["longitude"],
                "area_km2": result["area_km2"],
                "area_pixels": result["area_pixels"],
                "predicted_oil_percentage": result[
                    "predicted_oil_percentage"
                ],
                "mean_confidence": result["mean_confidence"],
                "max_confidence": result["max_confidence"],
                "crs": result["crs"],
                "geojson": str(output_path),
            })

            if index % 25 == 0 or index == len(prediction_files):
                print(
                    f"  Processed {index}/{len(prediction_files)}"
                )

    csv_path = OUTPUT_ROOT / "detection_metadata.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "scene_id",
            "category",
            "timestamp",
            "latitude",
            "longitude",
            "area_km2",
            "area_pixels",
            "predicted_oil_percentage",
            "mean_confidence",
            "max_confidence",
            "crs",
            "geojson",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n========================================")
    print("Detection output generation complete")
    print("========================================")
    print(f"GeoJSON: {GEOJSON_ROOT}")
    print(f"Metadata: {csv_path}")
    print(f"Total scenes: {len(rows)}")


if __name__ == "__main__":
    main()