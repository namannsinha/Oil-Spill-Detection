from pathlib import Path
import csv

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/api/detection", tags=["Detection"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "outputs" / "evaluation" / "scene_metrics.csv"


@router.get("/{scene}")
def get_detection(scene: str):
    """
    Return detection/evaluation information for a Part III scene.

    Example:
        GET /api/detection/00074
    """

    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="scene_metrics.csv not found"
        )

    requested_scene = scene

    if not requested_scene.endswith(".tif"):
        requested_scene += ".tif"

    with METRICS_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["scene"] == requested_scene:
                return {
                    "scene": row["scene"].replace(".tif", ""),
                    "category": row["category"],
                    "detection": {
                        "predicted_oil_percentage": float(
                            row["predicted_oil_percentage"]
                        ),
                        "actual_oil_percentage": float(
                            row["actual_oil_percentage"]
                        ),
                        "predicted_oil_pixels": int(
                            row["predicted_oil_pixels"]
                        ),
                        "actual_oil_pixels": int(
                            row["actual_oil_pixels"]
                        ),
                    },
                    "evaluation": {
                        "dice": float(row["dice"]),
                        "iou": float(row["iou"]),
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                    },
                    "confusion": {
                        "tp": int(row["tp"]),
                        "fp": int(row["fp"]),
                        "fn": int(row["fn"]),
                        "tn": int(row["tn"]),
                    },
                    "valid_pixels": int(row["valid_pixels"]),
                    "false_positive_scene": bool(
                        int(row["false_positive_scene"])
                    ),
                }

    raise HTTPException(
        status_code=404,
        detail=f"Scene '{scene}' not found"
    )