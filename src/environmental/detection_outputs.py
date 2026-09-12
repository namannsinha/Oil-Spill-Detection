from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def extract_spill_polygon(
    mask_path,
    probability_path=None,
    threshold=0.5,
    min_pixels=20,
):
    """
    Convert a binary prediction mask into geographic spill polygons.

    Parameters
    ----------
    mask_path : str or Path
        Binary prediction GeoTIFF.
    probability_path : str or Path, optional
        Probability GeoTIFF from U-Net.
    threshold : float
        Probability threshold used to create binary mask.
    min_pixels : int
        Remove tiny connected regions.

    Returns
    -------
    dict
        Spill geometry and metadata.
    """

    mask_path = Path(mask_path)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile

    mask = mask > 0

    if not np.any(mask):
        return {
            "geometry": None,
            "area_pixels": 0,
            "area": 0.0,
            "centroid": None,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "predicted_oil_percentage": 0.0,
        }

    geometries = []

    for geom, value in shapes(
        mask.astype(np.uint8),
        mask=mask,
        transform=transform,
    ):
        if value != 1:
            continue

        polygon = shape(geom)

        if polygon.is_empty:
            continue

        # Approximate pixel area filtering.
        if polygon.area <= min_pixels * abs(transform.a * transform.e):
            continue

        geometries.append(polygon)

    if not geometries:
        return {
            "geometry": None,
            "area_pixels": int(mask.sum()),
            "area": 0.0,
            "centroid": None,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "predicted_oil_percentage": (
                float(mask.mean() * 100.0)
            ),
        }

    geometry = unary_union(geometries)

    confidence = None

    if probability_path is not None:
        with rasterio.open(probability_path) as src:
            confidence = src.read(1).astype(np.float32)

    if confidence is not None:
        valid_confidence = confidence[mask]

        mean_confidence = float(
            np.mean(valid_confidence)
        )

        max_confidence = float(
            np.max(valid_confidence)
        )
    else:
        mean_confidence = 0.0
        max_confidence = 0.0

    centroid = geometry.centroid

    # Geographic CRS such as EPSG:4326:
    # centroid.x = longitude
    # centroid.y = latitude
    centroid_data = {
        "longitude": float(centroid.x),
        "latitude": float(centroid.y),
    }

    return {
        "geometry": mapping(geometry),
        "area_pixels": int(mask.sum()),
        "area": float(geometry.area),
        "centroid": centroid_data,
        "mean_confidence": mean_confidence,
        "max_confidence": max_confidence,
        "predicted_oil_percentage": (
            float(mask.mean() * 100.0)
        ),
        "crs": str(crs) if crs else None,
    }


def save_spill_geojson(
    result,
    output_path,
    properties=None,
):
    """
    Save extracted spill geometry as GeoJSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if result["geometry"] is None:
        features = []
    else:
        properties = properties or {}

        properties.update({
            "area_pixels": result["area_pixels"],
            "area": result["area"],
            "mean_confidence": result["mean_confidence"],
            "max_confidence": result["max_confidence"],
            "predicted_oil_percentage":
                result["predicted_oil_percentage"],
        })

        features = [{
            "type": "Feature",
            "geometry": result["geometry"],
            "properties": properties,
        }]

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            geojson,
            f,
            indent=2,
        )