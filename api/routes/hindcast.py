from pathlib import Path
import json

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/api/hindcast", tags=["Hindcast"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENTS_DIR = PROJECT_ROOT / "outputs" / "environmental" / "events"
EVENT_PAIR_FILE = EVENTS_DIR / "event_pair.json"


@router.get("/{event_id}")
def get_hindcast(event_id: str):
    """
    Return historical spill-event information and observed movement.

    Example:
        GET /api/hindcast/20171017_20171018_001
    """

    if not EVENT_PAIR_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="event_pair.json not found"
        )

    with EVENT_PAIR_FILE.open("r", encoding="utf-8") as f:
        event = json.load(f)

    if event["event_id"] != event_id:
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' not found"
        )

    t0 = event["t0"]
    t1 = event["t1"]

    return {
        "event_id": event["event_id"],

        "time_delta_hours": event["time_delta_hours"],

        "observed_movement": {
            "displacement_km": event[
                "observed_centroid_displacement_km"
            ],

            "from": {
                "timestamp": t0["timestamp"],
                "latitude": t0["centroid"]["latitude"],
                "longitude": t0["centroid"]["longitude"],
                "area_km2": t0["area_km2"],
                "scene_count": t0["scene_count"],
                "product_id": t0["product_id"],
            },

            "to": {
                "timestamp": t1["timestamp"],
                "latitude": t1["centroid"]["latitude"],
                "longitude": t1["centroid"]["longitude"],
                "area_km2": t1["area_km2"],
                "scene_count": t1["scene_count"],
                "product_id": t1["product_id"],
            },
        },

        "footprints": {
            "t0": t0["geojson"],
            "t1": t1["geojson"],
        },
    }