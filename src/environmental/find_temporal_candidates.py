from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "events"
    / "part1_product_groups.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "events"
    / "temporal_candidates.csv"
)


MIN_HOURS = 6
MAX_DAYS = 14
MAX_DISTANCE_KM = 100


def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0088

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(
        np.sqrt(a)
    )


def main():

    print("=" * 70)
    print("TEMPORAL OIL-SPILL CANDIDATE SEARCH")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)

    df["timestamp_start"] = pd.to_datetime(
        df["timestamp_start"]
    )

    candidates = []

    for i in range(len(df)):

        a = df.iloc[i]

        for j in range(i + 1, len(df)):

            b = df.iloc[j]

            time_delta = (
                b["timestamp_start"]
                - a["timestamp_start"]
            )

            hours = time_delta.total_seconds() / 3600.0

            if hours < MIN_HOURS:
                continue

            if hours > MAX_DAYS * 24:
                break

            distance = haversine_km(
                a["center_latitude"],
                a["center_longitude"],
                b["center_latitude"],
                b["center_longitude"],
            )

            if distance > MAX_DISTANCE_KM:
                continue

            # Require meaningful oil in both observations.
            if (
                a["oil_percent_max"] < 1.0
                or b["oil_percent_max"] < 1.0
            ):
                continue

            candidates.append({

                "product_a": a["product_id"],
                "product_b": b["product_id"],

                "timestamp_a": a["timestamp_start"].isoformat(),
                "timestamp_b": b["timestamp_start"].isoformat(),

                "time_difference_hours": hours,

                "distance_km": distance,

                "latitude_a": a["center_latitude"],
                "longitude_a": a["center_longitude"],

                "latitude_b": b["center_latitude"],
                "longitude_b": b["center_longitude"],

                "oil_percent_a": a["oil_percent_max"],
                "oil_percent_b": b["oil_percent_max"],

                "oil_area_km2_a": a["oil_area_km2_max"],
                "oil_area_km2_b": b["oil_area_km2_max"],

                "scene_count_a": a["scene_count"],
                "scene_count_b": b["scene_count"],

                "scene_ids_a": a["scene_ids"],
                "scene_ids_b": b["scene_ids"],
            })

    result = pd.DataFrame(candidates)

    if result.empty:

        print("No temporal candidates found.")

        return

    # ------------------------------------------------------------
    # Candidate score
    # ------------------------------------------------------------

    result["spatial_score"] = np.exp(
        -result["distance_km"] / 50.0
    )

    result["oil_score"] = np.minimum(
        result["oil_percent_a"],
        result["oil_percent_b"],
    ) / 10.0

    result["oil_score"] = result["oil_score"].clip(
        upper=1.0
    )

    result["candidate_score"] = (
        result["spatial_score"]
        * result["oil_score"]
    )

    result = result.sort_values(
        "candidate_score",
        ascending=False,
    ).reset_index(drop=True)

    result["rank"] = np.arange(
        1,
        len(result) + 1,
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 70)
    print("TOP TEMPORAL CANDIDATES")
    print("=" * 70)

    columns = [
        "rank",
        "timestamp_a",
        "timestamp_b",
        "time_difference_hours",
        "distance_km",
        "latitude_a",
        "longitude_a",
        "latitude_b",
        "longitude_b",
        "oil_percent_a",
        "oil_percent_b",
        "oil_area_km2_a",
        "oil_area_km2_b",
        "candidate_score",
    ]

    print(
        result.head(30)[columns].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(f"Candidates: {len(result)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()