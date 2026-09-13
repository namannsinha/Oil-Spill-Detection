from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from typing import List

from .models import AISPoint, VesselCandidate


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great-circle distance between two coordinates."""

    R = 6371.0

    lat1 = radians(lat1)
    lat2 = radians(lat2)
    dlat = lat2 - lat1

    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def calculate_relevance_score(
    distance_km: float,
    time_difference_hours: float,
) -> float:
    """
    Calculate a simple candidate relevance score.

    Closer vessels and vessels temporally closer to the
    spill detection receive higher scores.
    """

    distance_score = max(0.0, 1.0 - distance_km / 100.0)

    time_score = max(
        0.0,
        1.0 - time_difference_hours / 48.0,
    )

    score = 0.6 * distance_score + 0.4 * time_score

    return round(score, 4)


def find_candidate_vessels(
    vessels: List[AISPoint],
    spill_latitude: float,
    spill_longitude: float,
    detection_time: datetime,
    max_distance_km: float = 100.0,
    max_time_difference_hours: float = 48.0,
) -> List[VesselCandidate]:

    candidates = []

    for vessel in vessels:

        distance = haversine_km(
            spill_latitude,
            spill_longitude,
            vessel.latitude,
            vessel.longitude,
        )

        time_difference = abs(
            (vessel.timestamp - detection_time).total_seconds()
        ) / 3600.0

        if distance > max_distance_km:
            continue

        if time_difference > max_time_difference_hours:
            continue

        score = calculate_relevance_score(
            distance,
            time_difference,
        )

        candidates.append(
            VesselCandidate(
                mmsi=vessel.mmsi,
                vessel_type=vessel.vessel_type,
                latitude=vessel.latitude,
                longitude=vessel.longitude,
                timestamp=vessel.timestamp,
                distance_km=round(distance, 3),
                time_difference_hours=round(
                    time_difference,
                    3,
                ),
                relevance_score=score,
            )
        )

    candidates.sort(
        key=lambda x: x.relevance_score,
        reverse=True,
    )

    return candidates