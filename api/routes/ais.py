from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.environmental.ais.analyzer import find_candidate_vessels
from src.environmental.ais.demo_data import create_demo_ais_data


router = APIRouter(
    prefix="/api/ais",
    tags=["AIS"],
)


EVENTS = {
    "20171017_20171018_001": {
        "timestamp": datetime.fromisoformat(
            "2017-10-17T15:58:01.509662"
        ),
        "latitude": 34.77504313874554,
        "longitude": 30.372601197819165,
    }
}


@router.get("/{event_id}")
def get_ais_candidates(event_id: str):

    if event_id not in EVENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' not found",
        )

    event = EVENTS[event_id]

    # Existing synthetic/demo AIS data
    vessels = create_demo_ais_data()

    candidates = find_candidate_vessels(
        vessels,
        spill_latitude=event["latitude"],
        spill_longitude=event["longitude"],
        detection_time=event["timestamp"],
        max_distance_km=100.0,
        max_time_difference_hours=48.0,
    )

    return {
        "event_id": event_id,
        "data_source": "DEMO_SYNTHETIC_AIS",
        "warning": (
            "AIS candidates are synthetic demonstration data "
            "and must not be interpreted as real vessel attribution."
        ),
        "search_parameters": {
            "radius_km": 100,
            "time_window_hours": 48,
        },
        "candidates": [
            {
                "mmsi": vessel.mmsi,
                "vessel_type": vessel.vessel_type,
                "latitude": vessel.latitude,
                "longitude": vessel.longitude,
                "timestamp": vessel.timestamp.isoformat(),
                "distance_km": vessel.distance_km,
                "time_difference_hours": vessel.time_difference_hours,
                "relevance_score": vessel.relevance_score,
            }
            for vessel in candidates
        ],
    }