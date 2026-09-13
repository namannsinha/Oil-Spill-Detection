from datetime import datetime

from .models import AISPoint


def create_demo_ais_data():

    return [
        AISPoint(
            mmsi="DEMO-001",
            timestamp=datetime(2017, 10, 17, 14, 30),
            latitude=34.82,
            longitude=30.41,
            vessel_type="Tanker",
            speed_knots=11.2,
            course_deg=145.0,
        ),

        AISPoint(
            mmsi="DEMO-002",
            timestamp=datetime(2017, 10, 17, 18, 10),
            latitude=34.68,
            longitude=30.52,
            vessel_type="Cargo",
            speed_knots=14.7,
            course_deg=160.0,
        ),

        AISPoint(
            mmsi="DEMO-003",
            timestamp=datetime(2017, 10, 18, 1, 30),
            latitude=34.91,
            longitude=30.15,
            vessel_type="Fishing",
            speed_knots=5.4,
            course_deg=90.0,
        ),

        AISPoint(
            mmsi="DEMO-004",
            timestamp=datetime(2017, 10, 16, 8, 00),
            latitude=35.40,
            longitude=30.90,
            vessel_type="Cargo",
            speed_knots=13.1,
            course_deg=180.0,
        ),
    ]