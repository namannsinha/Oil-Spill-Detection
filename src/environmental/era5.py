"""
ERA5 environmental data provider.

Reads ERA5 hourly NetCDF data and returns 10 m wind
U/V components for a requested location and timestamp.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .environmental_data import EnvironmentalProvider, WindData


class ERA5Provider(EnvironmentalProvider):
    """
    Read ERA5 hourly wind data from a local NetCDF file.

    Expected variables:
        u10 -> 10 m eastward wind component
        v10 -> 10 m northward wind component
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"ERA5 dataset not found: {self.filepath}"
            )

        self.dataset = xr.open_dataset(self.filepath)

        required = {"u10", "v10"}

        missing = required - set(self.dataset.data_vars)

        if missing:
            raise ValueError(
                f"ERA5 dataset is missing variables: {sorted(missing)}"
            )

    def get_wind(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> WindData:

        ds = self.dataset

        # ERA5 longitude can be represented as either:
        #   0 ... 360
        # or
        #   -180 ... 180
        #
        # Convert requested longitude to the dataset convention.
        lon_values = ds["longitude"].values

        if np.min(lon_values) >= 0 and longitude < 0:
            longitude = longitude % 360

        elif np.max(lon_values) <= 180 and longitude > 180:
            longitude = ((longitude + 180) % 360) - 180

        point = ds.sel(
            latitude=latitude,
            longitude=longitude,
            time=timestamp,
            method="nearest",
        )

        u10 = float(point["u10"].values)
        v10 = float(point["v10"].values)

        return WindData(
            u10=u10,
            v10=v10,
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
        )

    def close(self):
        """Close the underlying NetCDF dataset."""

        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()