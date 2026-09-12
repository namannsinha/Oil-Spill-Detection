"""
Common interface for environmental data providers.

The environmental module uses:
    - wind U/V components
    - ocean current U/V components

Providers can later be backed by ERA5, Copernicus Marine,
local NetCDF files, APIs, etc.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class WindData:
    """Wind conditions at a specific location and time."""

    u10: float
    v10: float
    timestamp: datetime
    latitude: float
    longitude: float

    @property
    def speed(self) -> float:
        """Wind speed in m/s."""
        return (self.u10**2 + self.v10**2) ** 0.5


@dataclass
class CurrentData:
    """Ocean surface current at a specific location and time."""

    u: float
    v: float
    timestamp: datetime
    latitude: float
    longitude: float

    @property
    def speed(self) -> float:
        """Current speed in m/s."""
        return (self.u**2 + self.v**2) ** 0.5


@dataclass
class EnvironmentalData:
    """Combined environmental conditions."""

    wind: Optional[WindData] = None
    current: Optional[CurrentData] = None


class EnvironmentalProvider:
    """
    Base interface for environmental data providers.

    Concrete implementations will provide:
        get_wind()
        get_current()
    """

    def get_wind(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> WindData:
        raise NotImplementedError

    def get_current(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> CurrentData:
        raise NotImplementedError

    def get_environmental_data(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
    ) -> EnvironmentalData:

        wind = self.get_wind(
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,
        )

        current = self.get_current(
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,
        )

        return EnvironmentalData(
            wind=wind,
            current=current,
        )