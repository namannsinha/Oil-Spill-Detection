import cdsapi
from pathlib import Path

OUTPUT = Path(
    "outputs/environmental/event_2017_10_17/wind/era5_wind.nc"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

# 17 October: 15:00–23:00
client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": "2017",
        "month": "10",
        "day": "17",
        "time": [
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "area": [35.25, 30.00, 34.25, 30.75],
        "format": "netcdf",
    },
    str(OUTPUT.with_name("era5_wind_2017-10-17.nc")),
)

# 18 October: 00:00–04:00
client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": "2017",
        "month": "10",
        "day": "18",
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
        ],
        "area": [35.25, 30.00, 34.25, 30.75],
        "format": "netcdf",
    },
    str(OUTPUT.with_name("era5_wind_2017-10-18.nc")),
)

print("ERA5 wind download complete.")