import glob
import os
import xarray as xr

files = glob.glob(
    r"outputs\environmental\event_2017_10_17\wind\*.nc"
)

print("Files:", len(files))

for f in files:
    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(f))
    print("=" * 70)

    ds = xr.open_dataset(f)

    print(ds)

    print("u10 NaNs:", ds["u10"].isnull().sum().item())
    print("v10 NaNs:", ds["v10"].isnull().sum().item())

    print("Time:")
    print(ds.valid_time.values)
    
    print("Latitude:", ds.latitude.values)
    print("Longitude:", ds.longitude.values)

    ds.close()