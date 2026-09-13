from pathlib import Path

import pandas as pd
import numpy as np


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
    / "part1_product_groups.csv"
)


def main():

    print("=" * 70)
    print("PART I SENTINEL-1 PRODUCT GROUPING")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV)

    df["timestamp_start"] = pd.to_datetime(
        df["timestamp_start"]
    )

    print(f"Scenes loaded: {len(df)}")

    # ------------------------------------------------------------
    # Group by actual Sentinel-1 product
    # ------------------------------------------------------------

    groups = []

    for product_id, g in df.groupby(
        "product_id",
        dropna=False
    ):

        g = g.sort_values("timestamp_start")

        groups.append({
            "product_id": product_id,

            "scene_count": len(g),

            "timestamp_start": g["timestamp_start"].min().isoformat(),

            "timestamp_end": g["timestamp_start"].max().isoformat(),

            "duration_seconds": (
                g["timestamp_start"].max()
                - g["timestamp_start"].min()
            ).total_seconds(),

            "center_latitude": g["scene_latitude"].mean(),

            "center_longitude": g["scene_longitude"].mean(),

            "lat_min": g["scene_latitude"].min(),
            "lat_max": g["scene_latitude"].max(),

            "lon_min": g["scene_longitude"].min(),
            "lon_max": g["scene_longitude"].max(),

            "oil_percent_max": g["oil_percent"].max(),

            "oil_percent_mean": g["oil_percent"].mean(),

            "oil_area_km2_max": g["oil_area_km2"].max(),

            "oil_area_km2_sum": g["oil_area_km2"].sum(),

            "scene_ids": ",".join(
                g["scene_id"].astype(str)
            ),
        })

    result = pd.DataFrame(groups)

    # ------------------------------------------------------------
    # Rank products
    # ------------------------------------------------------------

    result = result.sort_values(
        [
            "oil_percent_max",
            "oil_area_km2_max",
        ],
        ascending=False,
    ).reset_index(drop=True)

    result["rank"] = np.arange(
        1,
        len(result) + 1
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ------------------------------------------------------------
    # Display
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 30 SENTINEL-1 PRODUCTS")
    print("=" * 70)

    columns = [
        "rank",
        "scene_count",
        "timestamp_start",
        "center_latitude",
        "center_longitude",
        "oil_percent_max",
        "oil_area_km2_max",
        "scene_ids",
        "product_id",
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

    print(f"Unique Sentinel-1 products: {len(result)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()