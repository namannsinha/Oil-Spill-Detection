import pandas as pd
from math import radians, sin, cos, asin, sqrt


MANIFEST = "data/events/part1_event_manifest.csv"


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * 6371 * asin(sqrt(a))


df = pd.read_csv(MANIFEST)

df["timestamp"] = pd.to_datetime(df["timestamp_start"])

# Only scenes containing meaningful oil
df = df[df["oil_percent"] >= 1].copy()

df = df.dropna(
    subset=[
        "oil_latitude",
        "oil_longitude",
        "oil_percent",
        "oil_area_km2",
    ]
)

records = df.to_dict("records")

candidates = []

for i in range(len(records)):
    for j in range(i + 1, len(records)):

        a = records[i]
        b = records[j]

        dt_hours = abs(
            (b["timestamp"] - a["timestamp"]).total_seconds()
        ) / 3600

        # 6 hours to 14 days
        if not (6 <= dt_hours <= 336):
            continue

        distance_km = haversine(
            a["oil_latitude"],
            a["oil_longitude"],
            b["oil_latitude"],
            b["oil_longitude"],
        )

        # Candidate spatial relationship
        if distance_km <= 300:

            candidates.append(
                {
                    "time_gap_hours": dt_hours,
                    "distance_km": distance_km,
                    "time_a": a["timestamp"],
                    "time_b": b["timestamp"],
                    "scene_a": a["image_path"],
                    "scene_b": b["image_path"],
                    "oil_percent_a": a["oil_percent"],
                    "oil_percent_b": b["oil_percent"],
                    "oil_area_a": a["oil_area_km2"],
                    "oil_area_b": b["oil_area_km2"],
                    "lat_a": a["oil_latitude"],
                    "lon_a": a["oil_longitude"],
                    "lat_b": b["oil_latitude"],
                    "lon_b": b["oil_longitude"],
                }
            )


result = pd.DataFrame(candidates)

print()
print("=" * 80)
print("TEMPORAL OIL-SPILL CANDIDATE SEARCH")
print("=" * 80)

print(f"\nOil-containing scenes: {len(records)}")
print(f"Candidates found: {len(result)}")

if len(result) == 0:
    print("\nNo candidates found.")
else:

    result = result.sort_values(
        ["distance_km", "time_gap_hours"]
    )

    print("\nTop 30 candidates:\n")
    

    for _, r in result.head(30).iterrows():

        print(
            f"{r['time_gap_hours']:7.1f} h | "
            f"{r['distance_km']:7.1f} km | "
            f"{r['time_a']} -> {r['time_b']} | "
            f"oil {r['oil_percent_a']:.2f}% -> "
            f"{r['oil_percent_b']:.2f}% | "
            f"area {r['oil_area_a']:.1f} -> "
            f"{r['oil_area_b']:.1f} km²"
        )

    output = "data/events/temporal_candidates_v2.csv"

    result.to_csv(output, index=False)

    print(f"\nSaved: {output}")
