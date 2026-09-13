"""
Stage 2 — Historical oil-spill particle hindcast.

T0 observed oil footprint
        ↓
particle release
        ↓
Copernicus current + ERA5 wind
        ↓
Lagrangian advection
        ↓
predicted T1 footprint
        ↓
comparison with observed T1 footprint
"""
from shapely.geometry import box
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin

from pathlib import Path
import json
import math

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
from pyproj import Geod


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

EVENTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "environmental"
    / "events"
)

ENVIRONMENT_FILE = EVENT_DIR / "hindcast_environment.nc"

T0_FOOTPRINT = EVENTS_DIR / "t0_footprint.geojson"
T1_FOOTPRINT = EVENTS_DIR / "t1_footprint.geojson"
EVENT_PAIR = EVENTS_DIR / "event_pair.json"

OUTPUT_DIR = EVENT_DIR / "hindcast"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

N_PARTICLES = 5000
RANDOM_SEED = 42

# Integration step.
# 10 minutes gives much smoother trajectories than
# integrating only once per hour.
STEP_MINUTES = 10

# Particle footprint density threshold.
# A grid cell is considered occupied if particles enter it.
FOOTPRINT_GRID_RESOLUTION_KM = 0.5

GEOD = Geod(ellps="WGS84")


# ============================================================
# Loading
# ============================================================

def load_environment():
    """Load prepared environmental forcing."""

    print("\nLoading environmental forcing...")
    print(f"File: {ENVIRONMENT_FILE}")

    if not ENVIRONMENT_FILE.exists():
        raise FileNotFoundError(ENVIRONMENT_FILE)

    ds = xr.open_dataset(ENVIRONMENT_FILE)

    required = {
        "spill_u",
        "spill_v",
    }

    missing = required - set(ds.data_vars)

    if missing:
        raise ValueError(
            f"Missing environmental variables: {sorted(missing)}"
        )

    print(ds)

    return ds


def load_event_pair():
    """Load T0/T1 event metadata."""

    print("\nLoading event metadata...")

    with open(EVENT_PAIR, "r", encoding="utf-8") as f:
        event = json.load(f)

    print(json.dumps(event, indent=2))

    return event


def load_footprint(path):
    """Load and union footprint geometry."""

    if not path.exists():
        raise FileNotFoundError(path)

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"Empty footprint: {path}")

    geometry = unary_union(gdf.geometry)

    if geometry.is_empty:
        raise ValueError(f"Invalid empty geometry: {path}")

    return geometry


# ============================================================
# Particle generation
# ============================================================

def generate_particles(footprint, n_particles, seed):
    """
    Generate random particles uniformly inside the T0 footprint.
    """

    print(
        f"\nGenerating {n_particles:,} particles..."
    )

    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = footprint.bounds

    particles = []

    # Rejection sampling.
    # The footprint is relatively small, so this is sufficient.
    attempts = 0

    while len(particles) < n_particles:

        remaining = n_particles - len(particles)

        # Generate a batch larger than necessary.
        batch_size = max(remaining * 3, 1000)

        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)

        for x, y in zip(xs, ys):

            point = Point(x, y)

            if footprint.contains(point):
                particles.append((x, y))

                if len(particles) >= n_particles:
                    break

        attempts += batch_size

        if attempts > n_particles * 1000:
            raise RuntimeError(
                "Particle generation is taking too long."
            )

    particles = np.asarray(
        particles,
        dtype=np.float64,
    )

    print(
        f"Generated particles: {len(particles):,}"
    )

    return particles


# ============================================================
# Environmental interpolation
# ============================================================

def interpolate_velocity(
    ds,
    latitudes,
    longitudes,
    timestamp,
):
    """
    Bilinearly interpolate spill velocity at particle positions.

    Returns:
        u, v in m/s
    """

    # xarray accepts vectorized latitude/longitude arrays.
    sampled = ds[["spill_u", "spill_v"]].interp(
        time=timestamp,
        latitude=xr.DataArray(
            latitudes,
            dims="particle",
        ),
        longitude=xr.DataArray(
            longitudes,
            dims="particle",
        ),
        method="linear",
    )

    u = sampled["spill_u"].values
    v = sampled["spill_v"].values

    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        invalid_count = int(
            np.sum(
                ~np.isfinite(u)
                | ~np.isfinite(v)
            )
        )

        raise ValueError(
            f"Environmental interpolation produced "
            f"{invalid_count} invalid particle velocities. "
            f"Particles have moved outside the environmental grid."
        )

    return u, v


# ============================================================
# Coordinate movement
# ============================================================

def move_particles(
    latitudes,
    longitudes,
    u,
    v,
    seconds,
):
    """
    Move particles using local east/north velocity.

    u = eastward velocity [m/s]
    v = northward velocity [m/s]

    Uses geodesic forward calculation.
    """

    distances = np.sqrt(u**2 + v**2) * seconds

    # Azimuth:
    # 0° = north
    # 90° = east
    azimuth = np.degrees(
        np.arctan2(u, v)
    )

    new_lon = np.empty_like(longitudes)
    new_lat = np.empty_like(latitudes)

    valid = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & np.isfinite(u)
        & np.isfinite(v)
    )

    new_lon[:] = longitudes
    new_lat[:] = latitudes

    if np.any(valid):

        lon2, lat2, _ = GEOD.fwd(
            longitudes[valid],
            latitudes[valid],
            azimuth[valid],
            distances[valid],
        )

        new_lon[valid] = lon2
        new_lat[valid] = lat2

    return new_lat, new_lon


# ============================================================
# Particle advection
# ============================================================

def advect_particles(
    ds,
    particles,
    start_time,
    end_time,
):
    """
    Advect particles from start_time to end_time.

    Uses midpoint/RK2 integration.
    """

    print("\nStarting particle advection...")

    current_time = np.datetime64(start_time)
    final_time = np.datetime64(end_time)

    step = np.timedelta64(
        STEP_MINUTES,
        "m",
    )

    latitudes = particles[:, 1].copy()
    longitudes = particles[:, 0].copy()

    initial_latitudes = latitudes.copy()
    initial_longitudes = longitudes.copy()

    total_seconds = (
        (
            final_time - current_time
        ) / np.timedelta64(1, "s")
    )

    step_seconds = STEP_MINUTES * 60

    n_steps = math.ceil(
        total_seconds / step_seconds
    )

    print(f"Integration steps: {n_steps}")
    print(f"Step size: {STEP_MINUTES} minutes")

    for step_index in range(n_steps):

        remaining_seconds = (
            final_time - current_time
        ) / np.timedelta64(1, "s")

        if remaining_seconds <= 0:
            break

        dt_seconds = min(
            step_seconds,
            float(remaining_seconds),
        )

        # ----------------------------------------------------
        # Velocity at beginning of step
        # ----------------------------------------------------

        u1, v1 = interpolate_velocity(
            ds,
            latitudes,
            longitudes,
            current_time,
        )

        # ----------------------------------------------------
        # Half-step
        # ----------------------------------------------------

        mid_lat, mid_lon = move_particles(
            latitudes,
            longitudes,
            u1,
            v1,
            dt_seconds / 2,
        )

        midpoint_time = (
            current_time
            + np.timedelta64(
                int(dt_seconds / 2),
                "s",
            )
        )

        # ----------------------------------------------------
        # Velocity at midpoint
        # ----------------------------------------------------

        u2, v2 = interpolate_velocity(
            ds,
            mid_lat,
            mid_lon,
            midpoint_time,
        )

        # ----------------------------------------------------
        # Full step
        # ----------------------------------------------------

        latitudes, longitudes = move_particles(
            latitudes,
            longitudes,
            u2,
            v2,
            dt_seconds,
        )

        current_time += np.timedelta64(
            int(dt_seconds),
            "s",
        )

        if (
            step_index % max(1, n_steps // 10)
            == 0
        ):
            progress = (
                100
                * step_index
                / n_steps
            )

            print(
                f"Progress: {progress:5.1f}%"
            )

    print("Particle advection complete.")

    initial = np.column_stack(
        [
            initial_longitudes,
            initial_latitudes,
        ]
    )

    final = np.column_stack(
        [
            longitudes,
            latitudes,
        ]
    )

    return initial, final


# ============================================================
# Geometry utilities
# ============================================================

def particle_multipoint(particles):
    """Create a MultiPoint geometry."""

    return MultiPoint(
        [
            Point(
                float(lon),
                float(lat),
            )
            for lon, lat in particles
        ]
    )


def create_particle_hull(particles):
    """
    Create a convex hull around predicted particles.

    This is useful as a first diagnostic but may overestimate
    the actual spill footprint.
    """

    points = particle_multipoint(particles)

    return points.convex_hull

def create_density_footprint(
    particles,
    resolution_km=0.5,
    min_particles_per_cell=1,
):
    """
    Create a predicted spill footprint from particle density.

    The study area is divided into approximately square
    geographic cells. A cell becomes part of the predicted
    footprint when it contains at least the requested number
    of particles.

    This avoids the strong area overestimation caused by a
    convex hull around dispersed particles.
    """

    print(
        f"\nCreating density-based footprint..."
    )
    print(
        f"Grid resolution: {resolution_km:.2f} km"
    )
    print(
        f"Minimum particles/cell: "
        f"{min_particles_per_cell}"
    )

    particles = np.asarray(
        particles,
        dtype=np.float64,
    )

    if particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError(
            "Particles must have shape (N, 2) "
            "with columns [longitude, latitude]."
        )

    if len(particles) == 0:
        raise ValueError(
            "No particles available."
        )

    # --------------------------------------------------------
    # Convert requested km resolution to approximate degrees
    # --------------------------------------------------------

    mean_lat = float(
        np.mean(particles[:, 1])
    )

    meters_per_degree_lat = 111_320.0

    meters_per_degree_lon = (
        111_320.0
        * np.cos(
            np.radians(mean_lat)
        )
    )

    resolution_lat = (
        resolution_km * 1000
        / meters_per_degree_lat
    )

    resolution_lon = (
        resolution_km * 1000
        / meters_per_degree_lon
    )

    # --------------------------------------------------------
    # Build grid
    # --------------------------------------------------------

    min_lon = float(
        particles[:, 0].min()
    )

    max_lon = float(
        particles[:, 0].max()
    )

    min_lat = float(
        particles[:, 1].min()
    )

    max_lat = float(
        particles[:, 1].max()
    )

    # Add one-cell margin.
    min_lon -= resolution_lon
    max_lon += resolution_lon
    min_lat -= resolution_lat
    max_lat += resolution_lat

    n_cols = int(
        np.ceil(
            (max_lon - min_lon)
            / resolution_lon
        )
    )

    n_rows = int(
        np.ceil(
            (max_lat - min_lat)
            / resolution_lat
        )
    )

    # --------------------------------------------------------
    # Count particles per cell
    # --------------------------------------------------------

    col = (
        (
            particles[:, 0]
            - min_lon
        )
        / resolution_lon
    ).astype(int)

    row = (
        (
            max_lat
            - particles[:, 1]
        )
        / resolution_lat
    ).astype(int)

    valid = (
        (row >= 0)
        & (row < n_rows)
        & (col >= 0)
        & (col < n_cols)
    )

    row = row[valid]
    col = col[valid]

    counts = np.zeros(
        (n_rows, n_cols),
        dtype=np.int32,
    )

    np.add.at(
        counts,
        (row, col),
        1,
    )

    occupied = (
        counts >= min_particles_per_cell
    )

    if not np.any(occupied):
        raise ValueError(
            "No occupied cells found."
        )

    # --------------------------------------------------------
    # Convert occupied cells to polygons
    # --------------------------------------------------------

    polygons = []

    for r, c in zip(
        *np.where(occupied)
    ):

        west = (
            min_lon
            + c * resolution_lon
        )

        east = (
            west
            + resolution_lon
        )

        north = (
            max_lat
            - r * resolution_lat
        )

        south = (
            north
            - resolution_lat
        )

        polygons.append(
            box(
                west,
                south,
                east,
                north,
            )
        )

    footprint = unary_union(
        polygons
    )

    print(
        f"Occupied cells: "
        f"{int(occupied.sum()):,}"
    )

    return footprint


def calculate_geodesic_area(geometry):
    """Calculate polygon area in km²."""

    if geometry.is_empty:
        return 0.0

    if geometry.geom_type == "Polygon":

        area, _ = GEOD.geometry_area_perimeter(
            geometry
        )

        return abs(area) / 1_000_000

    if geometry.geom_type == "MultiPolygon":

        total = 0.0

        for polygon in geometry.geoms:

            area, _ = GEOD.geometry_area_perimeter(
                polygon
            )

            total += abs(area)

        return total / 1_000_000

    return 0.0

def calculate_footprint_metrics(
    predicted,
    observed,
):
    """
    Compare predicted and observed spill footprints.

    Areas are calculated geodesically in km².
    """

    if predicted.is_empty:
        observed_area = calculate_geodesic_area(observed)

        return {
            "intersection_area_km2": 0.0,
            "union_area_km2": observed_area,
            "iou": 0.0,
            "dice": 0.0,
        }

    if observed.is_empty:
        predicted_area = calculate_geodesic_area(predicted)

        return {
            "intersection_area_km2": 0.0,
            "union_area_km2": predicted_area,
            "iou": 0.0,
            "dice": 0.0,
        }

    intersection = predicted.intersection(
        observed
    )

    union = predicted.union(
        observed
    )

    intersection_area = calculate_geodesic_area(
        intersection
    )

    union_area = calculate_geodesic_area(
        union
    )

    predicted_area = calculate_geodesic_area(
        predicted
    )

    observed_area = calculate_geodesic_area(
        observed
    )

    iou = (
        intersection_area / union_area
        if union_area > 0
        else 0.0
    )

    dice = (
        2.0 * intersection_area
        / (predicted_area + observed_area)
        if (predicted_area + observed_area) > 0
        else 0.0
    )

    return {
        "intersection_area_km2": intersection_area,
        "union_area_km2": union_area,
        "iou": iou,
        "dice": dice,
    }



def calculate_centroid(geometry):
    """Return latitude/longitude centroid."""

    centroid = geometry.centroid

    return {
        "latitude": centroid.y,
        "longitude": centroid.x,
    }


# ============================================================
# Distance / direction
# ============================================================

def geodesic_distance(
    lon1,
    lat1,
    lon2,
    lat2,
):
    """Distance between two coordinates in km."""

    _, _, distance = GEOD.inv(
        lon1,
        lat1,
        lon2,
        lat2,
    )

    return abs(distance) / 1000


def geodesic_bearing(
    lon1,
    lat1,
    lon2,
    lat2,
):
    """Initial bearing from point 1 to point 2."""

    azimuth, _, _ = GEOD.inv(
        lon1,
        lat1,
        lon2,
        lat2,
    )

    return (azimuth + 360) % 360


def angular_difference(a, b):
    """Smallest difference between two bearings."""

    difference = abs(a - b)

    return min(
        difference,
        360 - difference,
    )


# ============================================================
# Save GeoJSON
# ============================================================

def save_geometry(
    geometry,
    path,
    properties,
):
    """Save a geometry as GeoJSON."""

    gdf = gpd.GeoDataFrame(
        [properties],
        geometry=[geometry],
        crs="EPSG:4326",
    )

    gdf.to_file(
        path,
        driver="GeoJSON",
    )


def save_particles(
    particles,
    path,
):
    """Save particles as GeoJSON points."""

    geometry = [
        Point(
            float(lon),
            float(lat),
        )
        for lon, lat in particles
    ]

    gdf = gpd.GeoDataFrame(
        geometry=geometry,
        crs="EPSG:4326",
    )

    gdf.to_file(
        path,
        driver="GeoJSON",
    )


# ============================================================
# Visualization
# ============================================================

def create_visualization(
    t0_geometry,
    t1_geometry,
    predicted_geometry,
    initial_particles,
    final_particles,
    path,
):
    """Create T0 → predicted → observed T1 visualization."""

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    gpd.GeoSeries(
        [t0_geometry]
    ).plot(
        ax=ax,
        alpha=0.35,
        label="Observed T0",
    )

    gpd.GeoSeries(
        [predicted_geometry]
    ).plot(
        ax=ax,
        alpha=0.35,
        label="Predicted T1",
    )

    gpd.GeoSeries(
        [t1_geometry]
    ).plot(
        ax=ax,
        alpha=0.35,
        label="Observed T1",
    )

    ax.scatter(
        initial_particles[:, 0],
        initial_particles[:, 1],
        s=1,
        alpha=0.15,
        label="T0 particles",
    )

    ax.scatter(
        final_particles[:, 0],
        final_particles[:, 1],
        s=1,
        alpha=0.15,
        label="T1 particles",
    )

    ax.set_title(
        "Historical Oil-Spill Hindcast\n"
        "T0 → T1"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend()

    ax.set_aspect("equal")

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 2 — HISTORICAL PARTICLE HINDCAST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    ds = load_environment()

    event = load_event_pair()

    t0_geometry = load_footprint(
        T0_FOOTPRINT
    )

    t1_geometry = load_footprint(
        T1_FOOTPRINT
    )

    # --------------------------------------------------------
    # 2. Determine timestamps
    # --------------------------------------------------------

    # event_pair.json contains the event timestamps.
    # These keys may vary slightly depending on the
    # event discovery implementation, so inspect them.

    print("\nEvent metadata keys:")
    print(list(event.keys()))

    # Try common timestamp names.
    start_time = event["t0"]["timestamp"]
    end_time = event["t1"]["timestamp"]

    print(f"\nT0: {start_time}")
    print(f"T1: {end_time}")

    # Verify against the expected event duration.
    time_delta_hours = (
        (
            np.datetime64(end_time)
            - np.datetime64(start_time)
        )
        / np.timedelta64(1, "h")
    )

    print(
        f"Integration duration: "
        f"{float(time_delta_hours):.4f} hours"
    )   

    print(f"\nT0: {start_time}")
    print(f"T1: {end_time}")

    # --------------------------------------------------------
    # 3. Generate particles
    # --------------------------------------------------------

    particles_t0 = generate_particles(
        t0_geometry,
        N_PARTICLES,
        RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 4. Advect particles
    # --------------------------------------------------------

    particles_t0, particles_t1 = advect_particles(
        ds,
        particles_t0,
        start_time,
        end_time,
    )

    # --------------------------------------------------------
    # 5. Predicted footprint
    # --------------------------------------------------------

    # Keep convex hull as a diagnostic.
    convex_hull_geometry = create_particle_hull(
        particles_t1
    )

    # Primary predicted spill footprint.
    predicted_geometry = create_density_footprint(
        particles_t1,
        resolution_km=0.5,
        min_particles_per_cell=1,
    )

    # --------------------------------------------------------
    # 6. Centroids
    # --------------------------------------------------------

    t0_centroid = calculate_centroid(
        t0_geometry
    )

    t1_centroid = calculate_centroid(
        t1_geometry
    )

    predicted_centroid = calculate_centroid(
        predicted_geometry
    )

    # --------------------------------------------------------
    # 7. Movement metrics
    # --------------------------------------------------------

    observed_distance = geodesic_distance(
        t0_centroid["longitude"],
        t0_centroid["latitude"],
        t1_centroid["longitude"],
        t1_centroid["latitude"],
    )

    predicted_distance = geodesic_distance(
        t0_centroid["longitude"],
        t0_centroid["latitude"],
        predicted_centroid["longitude"],
        predicted_centroid["latitude"],
    )

    observed_bearing = geodesic_bearing(
        t0_centroid["longitude"],
        t0_centroid["latitude"],
        t1_centroid["longitude"],
        t1_centroid["latitude"],
    )

    predicted_bearing = geodesic_bearing(
        t0_centroid["longitude"],
        t0_centroid["latitude"],
        predicted_centroid["longitude"],
        predicted_centroid["latitude"],
    )

    centroid_error = geodesic_distance(
        predicted_centroid["longitude"],
        predicted_centroid["latitude"],
        t1_centroid["longitude"],
        t1_centroid["latitude"],
    )

    direction_error = angular_difference(
        predicted_bearing,
        observed_bearing,
    )

    # --------------------------------------------------------
    # 8. Footprint area
    # --------------------------------------------------------

    t0_area = calculate_geodesic_area(
        t0_geometry
    )

    t1_area = calculate_geodesic_area(
        t1_geometry
    )

    convex_hull_area = calculate_geodesic_area(
        convex_hull_geometry
    )

    predicted_area = calculate_geodesic_area(
        predicted_geometry
    )

    footprint_metrics = calculate_footprint_metrics(
        predicted_geometry,
        t1_geometry,
    )

    print(
        f"Convex hull area: "
        f"{convex_hull_area:.3f} km²"
    )

    print(
        f"Density footprint area: "
        f"{predicted_area:.3f} km²"
    )

    print(
        f"Footprint intersection: "
        f"{footprint_metrics['intersection_area_km2']:.3f} km²"
    )

    print(
        f"Footprint IoU: "
        f"{footprint_metrics['iou']:.4f}"
    )

    print(
        f"Footprint Dice: "
        f"{footprint_metrics['dice']:.4f}"
    )

    # --------------------------------------------------------
    # 9. Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HINDCAST RESULTS")
    print("=" * 70)

    print(
        f"T0 → T1 observed displacement: "
        f"{observed_distance:.3f} km"
    )

    print(
        f"T0 → predicted displacement: "
        f"{predicted_distance:.3f} km"
    )

    print(
        f"Centroid error: "
        f"{centroid_error:.3f} km"
    )

    print(
        f"Observed bearing: "
        f"{observed_bearing:.2f}°"
    )

    print(
        f"Predicted bearing: "
        f"{predicted_bearing:.2f}°"
    )

    print(
        f"Direction error: "
        f"{direction_error:.2f}°"
    )

    print(
        f"\nT0 area: "
        f"{t0_area:.3f} km²"
    )

    print(
        f"T1 observed area: "
        f"{t1_area:.3f} km²"
    )

    print(
        f"Predicted area: "
        f"{predicted_area:.3f} km²"
    )

    # --------------------------------------------------------
    # 10. Save particles
    # --------------------------------------------------------

    save_particles(
        particles_t0,
        OUTPUT_DIR / "particles_t0.geojson",
    )

    save_particles(
        particles_t1,
        OUTPUT_DIR / "particles_t1.geojson",
    )

    # --------------------------------------------------------
    # 11. Save predicted footprint
    # --------------------------------------------------------

    save_geometry(
        predicted_geometry,
        OUTPUT_DIR / "predicted_footprint.geojson",
        {
            "type": "predicted_t1_density",
            "resolution_km":
                FOOTPRINT_GRID_RESOLUTION_KM,
            "area_km2": predicted_area,
            "iou": footprint_metrics["iou"],
            "dice": footprint_metrics["dice"],
        },
    )

    save_geometry(
        convex_hull_geometry,
        OUTPUT_DIR / "predicted_convex_hull.geojson",
        {
            "type": "predicted_t1_convex_hull",
            "area_km2": convex_hull_area,
        },
    )

    # --------------------------------------------------------
    # 12. Save visualization
    # --------------------------------------------------------

    create_visualization(
        t0_geometry,
        t1_geometry,
        predicted_geometry,
        particles_t0,
        particles_t1,
        OUTPUT_DIR / "hindcast_comparison.png",
    )

    # --------------------------------------------------------
    # 13. Save metrics
    # --------------------------------------------------------

    metrics = {
        "t0_timestamp": str(start_time),
        "t1_timestamp": str(end_time),
        "integration_hours": (
            (
                np.datetime64(end_time)
                - np.datetime64(start_time)
            )
            / np.timedelta64(1, "h")
        ),
        "particles": N_PARTICLES,
        "step_minutes": STEP_MINUTES,
        "windage_coefficient": float(
            ds.wind_drift_coefficient.values
        ),
        "observed_displacement_km": observed_distance,
        "predicted_displacement_km": predicted_distance,
        "centroid_error_km": centroid_error,
        "observed_bearing_deg": observed_bearing,
        "predicted_bearing_deg": predicted_bearing,
        "direction_error_deg": direction_error,
        "t0_area_km2": t0_area,
        "t1_observed_area_km2": t1_area,
        "predicted_area_km2": predicted_area,

        "convex_hull_area_km2": convex_hull_area,

        "predicted_footprint_area_km2":
            predicted_area,

        "footprint_intersection_area_km2":
            footprint_metrics[
                "intersection_area_km2"
            ],

        "footprint_union_area_km2":
            footprint_metrics[
                "union_area_km2"
            ],

        "footprint_iou":
            footprint_metrics["iou"],

        "footprint_dice":
            footprint_metrics["dice"],
    }

    with open(
        OUTPUT_DIR / "hindcast_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)

    ds.close()

    print("\nStage 2 completed.")


if __name__ == "__main__":
    main()