from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class AISPoint:
    mmsi: str
    timestamp: datetime
    latitude: float
    longitude: float
    vessel_type: Optional[str] = None
    speed_knots: Optional[float] = None
    course_deg: Optional[float] = None


@dataclass
class VesselCandidate:
    mmsi: str
    vessel_type: Optional[str]
    latitude: float
    longitude: float
    timestamp: datetime
    distance_km: float
    time_difference_hours: float
    relevance_score: float