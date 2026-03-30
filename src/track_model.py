# %%
"""
track_model.py — FastF1 Track Data Extraction & Processing
═══════════════════════════════════════════════════════════
Stage 1 of MGU-K Thermal-Constrained ERS Deployment Optimiser

Extracts a race-representative lap from Monza via FastF1 using a
median-pace selection strategy (not the fastest lap — which is an
outlier unrepresentative of actual race thermal/deployment demands).
Resamples to a uniform 0.05 s time grid and classifies track segments
(straight vs corner) for downstream MPC cost-function tuning.

Falls back to a physics-based synthetic Monza profile when FastF1 is
unavailable (e.g. CI environments or offline development).

Author: Wolfgang Guio
Project: Ferrari F1 Engineering Academy 2026
"""

import os
import sys
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# %%
# ═══════════════════════════════════════════════════════════════════
# Data Container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrackData:
    """Uniform-time-step track data for a single lap.

    All arrays are length N (= lap_time / dt) and indexed by time step k.

    Attributes
    ----------
    t : np.ndarray          — time [s], shape (N,)
    s : np.ndarray          — cumulative distance [m], shape (N,)
    v : np.ndarray          — speed [m/s], shape (N,)
    throttle : np.ndarray   — throttle fraction [0, 1], shape (N,)
    brake : np.ndarray      — brake flag (0 or 1), shape (N,)
    gear : np.ndarray       — gear number, shape (N,)
    segment : np.ndarray    — 'straight' / 'corner' label, shape (N,)
    dt : float              — time-step size [s]
    lap_time : float        — total lap time [s]
    lap_distance : float    — total lap distance [m]
    source : str            — 'fastf1' or 'synthetic'
    """
    t: np.ndarray
    s: np.ndarray
    v: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    gear: np.ndarray
    segment: np.ndarray
    braking_zone: np.ndarray   # bool — True where extended MPC horizon applies
    dt: float
    lap_time: float
    lap_distance: float
    source: str

    @property
    def N(self) -> int:
        """Number of time steps."""
        return len(self.t)

    def summary(self) -> str:
        """Human-readable summary of the track data."""
        lines = [
            f"Track Data Summary ({self.source})",
            f"{'─' * 40}",
            f"  Lap time     : {self.lap_time:.2f} s",
            f"  Lap distance : {self.lap_distance:.0f} m",
            f"  Time step    : {self.dt} s",
            f"  Num steps    : {self.N}",
            f"  v_min        : {self.v.min():.1f} m/s ({self.v.min() * 3.6:.1f} km/h)",
            f"  v_max        : {self.v.max():.1f} m/s ({self.v.max() * 3.6:.1f} km/h)",
            f"  v_mean       : {self.v.mean():.1f} m/s ({self.v.mean() * 3.6:.1f} km/h)",
            f"  Straight %   : {100 * np.sum(self.segment == 'straight') / self.N:.1f}%",
            f"  Corner %     : {100 * np.sum(self.segment == 'corner') / self.N:.1f}%",
            f"  Braking zone%: {100 * np.sum(self.braking_zone) / self.N:.1f}%  (6 s horizon)",
        ]
        return "\n".join(lines)


# %%
# ═══════════════════════════════════════════════════════════════════
# Parameter Loading
# ═══════════════════════════════════════════════════════════════════

def load_params(params_path: Optional[str] = None) -> dict:
    """Load motor_params.yaml.

    Parameters
    ----------
    params_path : str, optional
        Explicit path; if None, looks relative to this file's location.

    Returns
    -------
    dict
    """
    if params_path is None:
        params_path = Path(__file__).resolve().parent.parent / "params" / "motor_params.yaml"
    else:
        params_path = Path(params_path)

    # Use UTF-8 (with optional BOM) so Unicode comments parse on Windows.
    with open(params_path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


# %%
# ═══════════════════════════════════════════════════════════════════
# Segment Classification
# ═══════════════════════════════════════════════════════════════════

def classify_segments(v: np.ndarray, dt: float,
                      accel_thresh: float = -5.0,
                      corner_speed_frac: float = 0.65) -> np.ndarray:
    """Classify each time step as 'straight' or 'corner'.

    A point is labelled 'corner' if:
      (a) longitudinal deceleration exceeds `accel_thresh` [m/s²], OR
      (b) speed is below `corner_speed_frac` × v_max.

    This is a simple heuristic; the MPC does not depend on it for
    constraint enforcement — it is used only for cost-weight
    scheduling (higher w_v on straights).

    Parameters
    ----------
    v           : speed profile [m/s]
    dt          : time step [s]
    accel_thresh: deceleration threshold for braking zones [m/s²]
    corner_speed_frac: fraction of v_max below which → corner

    Returns
    -------
    np.ndarray of str, same length as v
    """
    # Longitudinal acceleration via central differences
    a = np.gradient(v, dt)

    v_max = v.max()
    is_corner = (a < accel_thresh) | (v < corner_speed_frac * v_max)

    # Dilate corner labels by ±5 steps (~0.25 s) to capture entry/exit
    kernel = 5
    dilated = np.copy(is_corner)
    for i in range(len(is_corner)):
        lo = max(0, i - kernel)
        hi = min(len(is_corner), i + kernel + 1)
        if np.any(is_corner[lo:hi]):
            dilated[i] = True

    segment = np.where(dilated, "corner", "straight")
    return segment


def classify_braking_zones(v: np.ndarray, dt: float,
                           decel_thresh: float = -5.0,
                           pre_steps: int = 60,
                           post_steps: int = 20) -> np.ndarray:
    """Identify heavy braking zones for the variable-horizon MPC.

    Returns a boolean array that is True at every step where the extended
    prediction horizon (N_horizon_long, 6 s) should be used.

    The extended horizon activates BEFORE the braking event (pre_steps = 3 s
    by default) so the controller foresees the coming high-regen / high-
    thermal event and adjusts deployment proactively — not reactively.

    Criteria for a braking zone step
    ---------------------------------
    Deceleration < decel_thresh (default −5.0 m/s²) — captures the three
    principal Monza braking events: Turn 1 (Rettifilo), Lesmos complex,
    and Parabolica. Dilation adds a look-ahead pre-window and a tail window.

    Parameters
    ----------
    v            : speed profile [m/s]
    dt           : time step [s]
    decel_thresh : deceleration threshold [m/s²] — must be negative
    pre_steps    : steps BEFORE detected braking to extend horizon (look-ahead)
    post_steps   : steps AFTER detected braking to maintain extended horizon

    Returns
    -------
    braking_zone : np.ndarray of bool, shape (N,)
    """
    a = np.gradient(v, dt)
    is_heavy_braking = a < decel_thresh

    n = len(is_heavy_braking)
    braking_zone = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_heavy_braking[i]:
            lo = max(0, i - pre_steps)
            hi = min(n, i + post_steps + 1)
            braking_zone[lo:hi] = True

    return braking_zone


# %%
# ═══════════════════════════════════════════════════════════════════
# FastF1 Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_fastf1(year: int = 2025, session: str = "R",
                   circuit: str = "Monza", dt: float = 0.05,
                   cache_dir: str = "data/cache") -> TrackData:
    """Extract a race-representative lap from FastF1.

    Strategy: median-pace lap selection.
    ─────────────────────────────────────
    Rather than picking the fastest lap (which is an outlier — low fuel,
    fresh tyres, possibly a tow), we select the lap whose time is closest
    to the median of all clean race laps. This gives a speed profile that
    honestly represents the thermal loads and deployment demands the
    MGU-K faces during a typical race stint.

    Only "clean" laps are considered (no pit in/out, no safety car,
    no red flag, no deleted laps). If fewer than 5 clean laps remain
    after filtering, the fastest lap is used as a fallback.

    Parameters
    ----------
    year      : season year
    session   : session identifier ('R' = Race, 'Q' = Qualifying)
    circuit   : circuit name (must match FastF1 naming)
    dt        : target time step [s]
    cache_dir : FastF1 cache directory

    Returns
    -------
    TrackData
    """
    import fastf1
    import pandas as pd

    # Set up cache
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))

    # Load session
    print(f"[FastF1] Loading {year} {circuit} — session '{session}' ...")
    sess = fastf1.get_session(year, circuit, session)
    sess.load()

    # ── Filter for clean race laps ──────────────────────────────────
    all_laps = sess.laps
    # Keep only laps with valid lap times and telemetry
    clean = all_laps.pick_accurate()

    # Exclude pit in/out laps
    if "PitInTime" in clean.columns:
        clean = clean[clean["PitInTime"].isna() & clean["PitOutTime"].isna()]

    # Exclude laps under safety car or VSC if TrackStatus is available
    if "TrackStatus" in clean.columns:
        clean = clean[clean["TrackStatus"].isin(["1", 1, "1.0"])]

    # Require valid LapTime
    clean = clean.dropna(subset=["LapTime"])

    print(f"[FastF1] Total laps: {len(all_laps)}, clean laps: {len(clean)}")

    # ── Select median-pace lap ──────────────────────────────────────
    if len(clean) >= 5:
        lap_times = clean["LapTime"].dt.total_seconds()
        median_time = lap_times.median()
        # Pick the lap closest to median
        idx = (lap_times - median_time).abs().idxmin()
        selected = clean.loc[idx]
        selection_method = "median-pace"
        print(f"[FastF1] Median lap time: {median_time:.3f} s")
        print(f"[FastF1] Selected lap time: {selected['LapTime'].total_seconds():.3f} s "
              f"(Δ = {abs(selected['LapTime'].total_seconds() - median_time):.3f} s)")
    else:
        # Fallback: too few clean laps, use fastest
        selected = all_laps.pick_fastest()
        selection_method = "fastest (fallback)"
        print(f"[FastF1] Warning: only {len(clean)} clean laps — "
              f"falling back to fastest lap")

    tel = selected.get_telemetry()
    print(f"[FastF1] Driver: {selected['Driver']} | "
          f"Lap {selected.get('LapNumber', '?')} | "
          f"Selection: {selection_method}")

    # Extract raw arrays
    # FastF1 'Time' is a timedelta from session start; convert to seconds
    time_raw = tel["Time"].dt.total_seconds().values
    time_raw = time_raw - time_raw[0]  # zero-based

    speed_raw = tel["Speed"].values / 3.6      # km/h → m/s
    dist_raw  = tel["Distance"].values          # metres
    throttle_raw = tel["Throttle"].values / 100 # percentage → fraction
    brake_raw = tel["Brake"].values.astype(float)
    gear_raw  = tel["nGear"].values.astype(float)

    # Resample to uniform dt grid
    lap_time = time_raw[-1]
    t_uniform = np.arange(0, lap_time, dt)

    v = np.interp(t_uniform, time_raw, speed_raw)
    s = np.interp(t_uniform, time_raw, dist_raw)
    throttle = np.interp(t_uniform, time_raw, throttle_raw)
    brake = np.interp(t_uniform, time_raw, brake_raw)
    gear = np.interp(t_uniform, time_raw, gear_raw)
    gear = np.round(gear).astype(int)

    # Classify segments and braking zones
    segment = classify_segments(v, dt)
    braking_zone = classify_braking_zones(v, dt)

    return TrackData(
        t=t_uniform, s=s, v=v,
        throttle=throttle, brake=brake, gear=gear,
        segment=segment,
        braking_zone=braking_zone,
        dt=dt,
        lap_time=lap_time,
        lap_distance=dist_raw[-1],
        source="fastf1",
    )


# %%
# ═══════════════════════════════════════════════════════════════════
# Synthetic Monza Profile (fallback)
# ═══════════════════════════════════════════════════════════════════

def _build_synthetic_monza(dt: float = 0.05) -> TrackData:
    """Generate a physics-informed synthetic Monza speed profile.

    Based on Autodromo Nazionale di Monza layout:
      - Lap distance:  5793 m
      - Typical lap time: ~80 s (2025 race pace)
      - Key features: long straights, three chicanes, Lesmo curves,
        Ascari chicane, Parabolica

    The profile is constructed by defining speed targets at known
    track positions and interpolating with a smooth acceleration/
    braking model.

    Returns
    -------
    TrackData
    """
    lap_distance = 5793.0  # metres (Monza circuit length)

    # ── Define waypoints: (distance [m], speed [km/h]) ──
    # Based on published GPS traces and onboard data
    waypoints = np.array([
        # Start/finish straight
        [0,     340],
        [200,   345],
        # T1 Rettifilo chicane (braking zone)
        [600,   350],
        [700,   140],   # chicane apex 1
        [780,   120],   # chicane apex 2
        [900,   200],   # exit
        # Run to Curva Grande
        [1100,  310],
        [1400,  300],   # Curva Grande (slight lift)
        [1550,  280],
        # Straight to second chicane (Variante della Roggia)
        [1800,  330],
        [2050,  340],
        [2200,  145],   # chicane apex 1
        [2280,  125],   # chicane apex 2
        [2400,  200],   # exit
        # Run to Lesmo 1
        [2600,  290],
        [2800,  200],   # Lesmo 1 apex
        [2950,  230],   # exit
        # Lesmo 2
        [3100,  260],
        [3200,  195],   # Lesmo 2 apex
        [3350,  220],   # exit
        # Run down to Ascari
        [3600,  310],
        [3800,  320],
        [3950,  170],   # Ascari entry
        [4050,  155],   # Ascari apex
        [4200,  180],   # Ascari exit
        # Long straight to Parabolica
        [4500,  330],
        [4800,  340],
        [5050,  345],
        # Parabolica
        [5200,  200],   # Parabolica entry
        [5350,  170],   # Parabolica apex
        [5500,  220],   # Parabolica exit
        # Main straight (back to start/finish)
        [5793,  340],
    ])

    dist_wp = waypoints[:, 0]
    speed_wp = waypoints[:, 1] / 3.6  # → m/s

    # ── Create fine distance grid ──
    ds = 1.0  # 1 m resolution
    s_fine = np.arange(0, lap_distance, ds)

    # Interpolate speed vs distance (cubic-like via numpy)
    # Use piecewise linear on waypoints, then smooth
    v_fine = np.interp(s_fine, dist_wp, speed_wp)

    # Apply Gaussian smoothing to remove sharp kinks
    # (mimics real car acceleration/deceleration limits)
    kernel_size = 31
    kernel = np.exp(-0.5 * np.linspace(-3, 3, kernel_size)**2)
    kernel /= kernel.sum()
    v_smooth = np.convolve(v_fine, kernel, mode="same")

    # Fix boundary effects
    v_smooth[:kernel_size] = v_fine[:kernel_size]
    v_smooth[-kernel_size:] = v_fine[-kernel_size:]

    # ── Convert distance-domain to time-domain ──
    # dt_i = ds / v_i → cumulative sum gives time at each distance point
    dt_per_step = ds / np.maximum(v_smooth, 10.0)  # floor at 10 m/s
    t_fine = np.cumsum(dt_per_step)
    t_fine = t_fine - t_fine[0]  # zero-based

    lap_time = t_fine[-1]

    # ── Resample to uniform time grid ──
    t_uniform = np.arange(0, lap_time, dt)
    v_uniform = np.interp(t_uniform, t_fine, v_smooth)
    s_uniform = np.interp(t_uniform, t_fine, s_fine)

    # Ensure minimum speed (physical: car never stops on track)
    v_uniform = np.maximum(v_uniform, 25.0)  # ~90 km/h minimum

    # ── Derive throttle and brake from acceleration ──
    a = np.gradient(v_uniform, dt)
    throttle = np.clip(a / 15.0 + 0.5, 0.0, 1.0)   # heuristic mapping
    brake = (a < -5.0).astype(float)

    # ── Estimate gear from speed (8-speed gearbox, ~45 km/h per gear) ──
    gear = np.clip(np.floor(v_uniform * 3.6 / 45.0) + 1, 1, 8).astype(int)

    # ── Classify segments and braking zones ──
    segment = classify_segments(v_uniform, dt)
    braking_zone = classify_braking_zones(v_uniform, dt)
    n_bz = int(np.sum(braking_zone))

    print(f"[Synthetic] Monza profile generated: {lap_time:.1f} s, "
          f"{lap_distance:.0f} m, {len(t_uniform)} steps "
          f"({n_bz} braking-zone steps = {n_bz * dt:.1f} s @ 6 s horizon)")

    return TrackData(
        t=t_uniform, s=s_uniform, v=v_uniform,
        throttle=throttle, brake=brake, gear=gear,
        segment=segment,
        braking_zone=braking_zone,
        dt=dt,
        lap_time=lap_time,
        lap_distance=lap_distance,
        source="synthetic",
    )


# %%
# ═══════════════════════════════════════════════════════════════════
# Main Public Interface
# ═══════════════════════════════════════════════════════════════════

def load_track(params: Optional[dict] = None,
               force_synthetic: bool = False) -> TrackData:
    """Load Monza track data — FastF1 if available, synthetic fallback.

    Parameters
    ----------
    params : dict, optional
        Parsed motor_params.yaml; if None, loads automatically.
    force_synthetic : bool
        If True, skip FastF1 and use synthetic profile directly.

    Returns
    -------
    TrackData
    """
    if params is None:
        params = load_params()

    track_cfg = params["track"]
    mpc_cfg = params["mpc"]
    dt = mpc_cfg["dt"]

    if not force_synthetic:
        try:
            return extract_fastf1(
                year=track_cfg["year"],
                session=track_cfg["session"],
                circuit=track_cfg["circuit"],
                dt=dt,
                cache_dir=track_cfg["cache_dir"],
            )
        except Exception as e:
            print(f"[track_model] FastF1 unavailable ({e}), "
                  f"falling back to synthetic profile.")

    return _build_synthetic_monza(dt=dt)


# %%
# ═══════════════════════════════════════════════════════════════════
# Plotting Utilities
# ═══════════════════════════════════════════════════════════════════

def plot_speed_profile(track: TrackData, save_path: Optional[str] = None,
                       show: bool = True):
    """Plot speed vs distance with segment colouring.

    Parameters
    ----------
    track     : TrackData instance
    save_path : if given, save figure to this path
    show      : whether to call plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"Monza Lap Profile ({track.source})", fontsize=14, fontweight="bold")

    # ── Speed vs distance with segment shading ──
    ax = axes[0]
    s_km = track.s / 1000

    # Shade corners
    is_corner = track.segment == "corner"
    corner_start = None
    for i in range(len(is_corner)):
        if is_corner[i] and corner_start is None:
            corner_start = i
        elif not is_corner[i] and corner_start is not None:
            ax.axvspan(s_km[corner_start], s_km[i - 1],
                       alpha=0.15, color="red", zorder=0)
            corner_start = None
    if corner_start is not None:
        ax.axvspan(s_km[corner_start], s_km[-1],
                   alpha=0.15, color="red", zorder=0)

    ax.plot(s_km, track.v * 3.6, color="#DC0000", linewidth=1.2, label="Speed")
    ax.set_ylabel("Speed [km/h]")
    ax.set_ylim(0, 400)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    straight_patch = mpatches.Patch(color="white", label="Straight")
    corner_patch = mpatches.Patch(color="red", alpha=0.15, label="Corner")
    ax.legend(handles=[straight_patch, corner_patch], loc="upper left")

    # ── Throttle / Brake ──
    ax = axes[1]
    ax.fill_between(s_km, 0, track.throttle, color="#00A000", alpha=0.6, label="Throttle")
    ax.fill_between(s_km, 0, -track.brake, color="#DC0000", alpha=0.6, label="Brake")
    ax.set_ylabel("Throttle / Brake")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # ── Gear ──
    ax = axes[2]
    ax.step(s_km, track.gear, color="#0050A0", linewidth=1.0, where="mid")
    ax.set_ylabel("Gear")
    ax.set_xlabel("Distance [km]")
    ax.set_ylim(0, 9)
    ax.set_yticks(range(1, 9))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_speed_vs_time(track: TrackData, save_path: Optional[str] = None,
                       show: bool = True):
    """Plot speed vs time — the view the MPC actually sees.

    Parameters
    ----------
    track     : TrackData instance
    save_path : if given, save figure to this path
    show      : whether to call plt.show()
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(track.t, track.v * 3.6, color="#DC0000", linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title(f"Monza Speed vs Time ({track.source})", fontweight="bold")
    ax.set_ylim(0, 400)
    ax.grid(True, alpha=0.3)

    # Annotate key stats
    textstr = (f"Lap time: {track.lap_time:.2f} s\n"
               f"v_max: {track.v.max() * 3.6:.0f} km/h\n"
               f"v_avg: {track.v.mean() * 3.6:.0f} km/h\n"
               f"Steps: {track.N}")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# %%
# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Stage 1: Track Model — Monza Data Extraction")
    print("=" * 60)

    # Load parameters
    params = load_params()
    print(f"\n[params] Loaded motor_params.yaml")

    # Load track data
    track = load_track(params)
    print(f"\n{track.summary()}")

    # Generate plots
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plot_speed_profile(track, save_path=str(plots_dir / "monza_speed_profile.png"),
                       show=False)
    plot_speed_vs_time(track, save_path=str(plots_dir / "monza_speed_vs_time.png"),
                       show=False)

    print(f"\n[Done] Plots saved to {plots_dir}/")
