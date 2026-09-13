from datetime import datetime

from .analyzer import find_candidate_vessels
from .demo_data import create_demo_ais_data


def main():

    vessels = create_demo_ais_data()

    detection_time = datetime(
        2017,
        10,
        17,
        15,
        58,
    )

    candidates = find_candidate_vessels(
        vessels=vessels,
        spill_latitude=34.775043,
        spill_longitude=30.372601,
        detection_time=detection_time,
    )

    print("\nAIS CANDIDATE ANALYSIS")
    print("=" * 50)

    for candidate in candidates:

        print(
            f"MMSI: {candidate.mmsi}\n"
            f"Type: {candidate.vessel_type}\n"
            f"Distance: {candidate.distance_km} km\n"
            f"Time difference: "
            f"{candidate.time_difference_hours} hours\n"
            f"Relevance score: "
            f"{candidate.relevance_score}\n"
            f"{'-' * 50}"
        )


if __name__ == "__main__":
    main()