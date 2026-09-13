from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "events"
    / "part1_event_manifest.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "events"
    / "part1_event_groups.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Scenes acquired within this many seconds are treated as
# belonging to the same acquisition batch.
TIME_THRESHOLD_SECONDS = 10

# Spatial grouping threshold in degrees.
#
# This is deliberately conservative. We will inspect the
# resulting groups before making scientific conclusions.
SPATIAL_THRESHOLD_DEGREES = 0.5


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate approximate great-circle distance in km.
    """

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

    return 2 * R * np.arcsin(np.sqrt(a))


# ============================================================
# GROUPING
# ============================================================

def build_groups(df):

    df = df.copy()

    df["timestamp_start"] = pd.to_datetime(
        df["timestamp_start"]
    )

    df = df.sort_values(
        "timestamp_start"
    ).reset_index(drop=True)

    groups = []

    current_group = []
    previous_time = None

    for idx, row in df.iterrows():

        current_time = row["timestamp_start"]

        if previous_time is None:

            current_group = [idx]

        else:

            time_difference = (
                current_time - previous_time
            ).total_seconds()

            if time_difference <= TIME_THRESHOLD_SECONDS:

                current_group.append(idx)

            else:

                groups.append(current_group)

                current_group = [idx]

        previous_time = current_time

    if current_group:
        groups.append(current_group)

    return groups


# ============================================================
# GROUP SUMMARY
# ============================================================

def summarize_group(group_id, indices, df):

    g = df.iloc[indices]
    g["timestamp_start"] = pd.to_datetime(
        g["timestamp_start"]
    )

    lat_min = g["scene_latitude"].min()
    lat_max = g["scene_latitude"].max()

    lon_min = g["scene_longitude"].min()
    lon_max = g["scene_longitude"].max()

    center_lat = g["scene_latitude"].mean()
    center_lon = g["scene_longitude"].mean()

    timestamp_min = g["timestamp_start"].min()
    timestamp_max = g["timestamp_start"].max()

    duration_seconds = (
        timestamp_max - timestamp_min
    ).total_seconds()

    # Approximate spatial extent.
    north_south_km = haversine_km(
        lat_min,
        center_lon,
        lat_max,
        center_lon,
    )

    east_west_km = haversine_km(
        center_lat,
        lon_min,
        center_lat,
        lon_max,
    )

    return {
        "group_id": group_id,

        "scene_count": len(g),

        "timestamp_start": timestamp_min.isoformat(),
        "timestamp_end": timestamp_max.isoformat(),

        "duration_seconds": duration_seconds,

        "center_latitude": center_lat,
        "center_longitude": center_lon,

        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,

        "spatial_height_km": north_south_km,
        "spatial_width_km": east_west_km,

        "oil_percent_max": g["oil_percent"].max(),
        "oil_percent_mean": g["oil_percent"].mean(),

        "oil_area_km2_sum": g["oil_area_km2"].sum(),
        "oil_area_km2_max": g["oil_area_km2"].max(),

        "scene_ids": ",".join(
            g["scene_id"].astype(str)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PART I EVENT GROUPING")
    print("=" * 70)

    print(f"Input : {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    df = pd.read_csv(INPUT_CSV)

    print(f"Scenes loaded: {len(df)}")

    groups = build_groups(df)

    print(f"Initial acquisition groups: {len(groups)}")
    print()

    summaries = []

    for group_id, indices in enumerate(groups, start=1):

        summary = summarize_group(
            group_id,
            indices,
            df,
        )

        summaries.append(summary)

    result = pd.DataFrame(summaries)

    # Rank primarily by maximum oil percentage.
    result = result.sort_values(
        [
            "oil_percent_max",
            "oil_area_km2_max",
        ],
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

    print("=" * 70)
    print("TOP 30 ACQUISITION GROUPS")
    print("=" * 70)

    columns = [
        "rank",
        "group_id",
        "scene_count",
        "timestamp_start",
        "center_latitude",
        "center_longitude",
        "spatial_height_km",
        "spatial_width_km",
        "oil_percent_max",
        "oil_area_km2_max",
        "scene_ids",
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

    print(f"Groups : {len(result)}")
    print(f"Output : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()