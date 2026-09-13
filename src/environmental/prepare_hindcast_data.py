from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "environmental"
    / "event_2017_10_17"
)

CURRENT_FILE = next(EVENT_DIR.glob("cmems_mod_med_phy-cur_my_4.2km_PT1H-m_*.nc"))

WIND_DIR = EVENT_DIR / "wind"

WIND_FILES = [
    WIND_DIR / "era5_wind_2017-10-17.nc",
    WIND_DIR / "era5_wind_2017-10-18.nc",
]

OUTPUT_FILE = EVENT_DIR / "hindcast_environment.nc"


# ============================================================
# Configuration
# ============================================================

# Initial baseline wind-drift coefficient.
# This is NOT calibrated yet.
WIND_DRIFT_COEFFICIENT = 0.03

START_TIME = "2017-10-17T15:00:00"
END_TIME = "2017-10-18T04:00:00"


# ============================================================
# Helper functions
# ============================================================

def load_currents():
    """Load Copernicus hourly surface currents."""

    print("\nLoading Copernicus currents...")
    print(f"File: {CURRENT_FILE}")

    ds = xr.open_dataset(CURRENT_FILE)

    # Keep only the hindcast period.
    ds = ds.sel(
        time=(
            (ds.time >= np.datetime64(START_TIME))
            & (ds.time <= np.datetime64(END_TIME))
        )
    )

    required = ["uo", "vo"]

    for variable in required:
        if variable not in ds:
            raise ValueError(
                f"Missing current variable: {variable}"
            )

    print(
        f"Current time steps: {ds.sizes['time']}"
    )

    print(
        f"Current grid: "
        f"{ds.sizes['latitude']} latitude × "
        f"{ds.sizes['longitude']} longitude"
    )

    return ds


def load_wind():
    print("\nLoading ERA5 wind...")

    datasets = []

    for path in WIND_FILES:
        print(f"File: {path}")

        ds = xr.open_dataset(path)

        # ERA5 uses valid_time rather than time
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        # Keep only required variables
        ds = ds[["u10", "v10"]]

        # Normalize coordinate ordering
        ds = ds.sortby("latitude")
        ds = ds.sortby("longitude")
        ds = ds.sortby("time")

        datasets.append(ds)

    # Make sure both files use exactly the same spatial grid
    wind = xr.concat(
        datasets,
        dim="time",
        join="exact"
    )

    wind = wind.sortby("time")

    print(f"Wind time steps: {wind.sizes['time']}")

    return wind


def interpolate_wind_to_current_grid(
    wind,
    currents,
):
    """
    Interpolate ERA5 wind spatially onto the
    higher-resolution Copernicus current grid.
    """

    print(
        "\nInterpolating ERA5 wind "
        "onto current grid..."
    )

    # ERA5 latitude is descending.
    # xarray interpolation works more reliably
    # when coordinates are monotonic ascending.
    wind = wind.sortby("latitude")
    wind = wind.sortby("longitude")

    target_latitude = currents.latitude
    target_longitude = currents.longitude

    wind_interp = wind.interp(
        latitude=target_latitude,
        longitude=target_longitude,
        method="linear",
    )

    return wind_interp


def interpolate_wind_in_time(
    wind,
    target_times,
):
    """
    Interpolate ERA5 hourly wind onto the exact
    Copernicus current timestamps.
    """

    print("\nInterpolating wind in time...")

    if "valid_time" in wind.coords:
        wind = wind.rename({"valid_time": "time"})

    wind = wind.sortby("time")

    wind = wind.interp(
        time=target_times,
        method="linear",
    )

    return wind


def validate_data(
    currents,
    wind,
):
    """Run sanity checks before saving."""

    print("\nRunning validation...")

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    if not np.array_equal(
        currents.time.values,
        wind.time.values,
    ):
        raise ValueError(
            "Current and wind time coordinates "
            "do not match."
        )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    variables = [
        currents.uo,
        currents.vo,
        wind.u10,
        wind.v10,
    ]

    names = [
        "uo",
        "vo",
        "u10",
        "v10",
    ]

    for name, data in zip(names, variables):
        nan_count = int(
            data.isnull().sum().item()
        )

        print(
            f"{name} NaNs: {nan_count}"
        )

        if nan_count > 0:
            raise ValueError(
                f"{name} contains NaN values."
            )

    # --------------------------------------------------------
    # Basic ranges
    # --------------------------------------------------------

    print(
        f"uo range: "
        f"{float(currents.uo.min()):.4f} → "
        f"{float(currents.uo.max()):.4f} m/s"
    )

    print(
        f"vo range: "
        f"{float(currents.vo.min()):.4f} → "
        f"{float(currents.vo.max()):.4f} m/s"
    )

    print(
        f"u10 range: "
        f"{float(wind.u10.min()):.4f} → "
        f"{float(wind.u10.max()):.4f} m/s"
    )

    print(
        f"v10 range: "
        f"{float(wind.v10.min()):.4f} → "
        f"{float(wind.v10.max()):.4f} m/s"
    )

    print("Validation passed.")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING HISTORICAL HINDCAST ENVIRONMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load current
    # --------------------------------------------------------

    currents = load_currents()

    # --------------------------------------------------------
    # 2. Load ERA5 wind
    # --------------------------------------------------------

    wind = load_wind()

    # --------------------------------------------------------
    # 3. Spatial interpolation
    # --------------------------------------------------------

    wind = interpolate_wind_to_current_grid(
        wind,
        currents,
    )

    # --------------------------------------------------------
    # 4. Temporal interpolation
    # --------------------------------------------------------

    wind = interpolate_wind_in_time(
        wind,
        currents.time,
    )

    # --------------------------------------------------------
    # 5. Validation
    # --------------------------------------------------------

    validate_data(
        currents,
        wind,
    )

    # --------------------------------------------------------
    # 6. Create combined dataset
    # --------------------------------------------------------

    print(
        "\nCreating combined environmental dataset..."
    )

    combined = xr.Dataset(
        data_vars={
            "uo": currents.uo,
            "vo": currents.vo,
            "u10": wind.u10,
            "v10": wind.v10,
        },
        coords={
            "time": currents.time,
            "latitude": currents.latitude,
            "longitude": currents.longitude,
        },
    )

    # --------------------------------------------------------
    # 7. Calculate baseline spill velocity
    # --------------------------------------------------------

    combined["spill_u"] = (
        combined["uo"]
        + WIND_DRIFT_COEFFICIENT
        * combined["u10"]
    )

    combined["spill_v"] = (
        combined["vo"]
        + WIND_DRIFT_COEFFICIENT
        * combined["v10"]
    )

    combined["wind_drift_coefficient"] = (
        WIND_DRIFT_COEFFICIENT
    )

    # --------------------------------------------------------
    # 8. Metadata
    # --------------------------------------------------------

    combined.attrs.update(
        {
            "description": (
                "Environmental forcing for "
                "historical oil-spill hindcast."
            ),
            "current_source": (
                "Copernicus Mediterranean Sea "
                "Physics Reanalysis"
            ),
            "current_dataset": (
                "cmems_mod_med_phy-cur_my_4.2km_PT1H-m"
            ),
            "wind_source": (
                "ERA5 hourly reanalysis"
            ),
            "wind_variables": (
                "10m_u_component_of_wind, "
                "10m_v_component_of_wind"
            ),
            "wind_drift_coefficient": (
                WIND_DRIFT_COEFFICIENT
            ),
            "wind_drift_note": (
                "Initial baseline assumption; "
                "not calibrated."
            ),
            "hindcast_start": START_TIME,
            "hindcast_end": END_TIME,
        }
    )

    # --------------------------------------------------------
    # 9. Save
    # --------------------------------------------------------

    print(
        f"\nSaving:\n{OUTPUT_FILE}"
    )

    combined.to_netcdf(
        OUTPUT_FILE
    )

    print(
        "\nHindcast environmental preparation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()