# %%
"""
main.py — MGU-K Thermal-Constrained ERS Deployment Optimiser
═════════════════════════════════════════════════════════════
Stage 5: Couples all modules and runs the full-lap MPC simulation.

This is the top-level script that:
  1. Loads track data (Monza, median-pace race lap)
  2. Initialises the PMSM loss model, thermal network, and MPC controller
  3. Runs the full ~80 s lap as a closed-loop MPC simulation
  4. Generates all Stage 6 output plots

The central insight demonstrated: deployment strategy and thermal
management are the SAME problem — the MPC must simultaneously
maximise speed and prevent magnet demagnetisation above 140°C.

Author: Wolfgang Guio
Project: Ferrari F1 Engineering Academy 2026
"""

import sys

# Force output to UTF-8 to handle box-drawing characters on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
import yaml
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.track_model import load_track, load_params, TrackData
from src.pmsm_losses import (
    load_motor_params, MotorParams, total_losses,
    copper_losses, iron_losses, power_to_current, speed_to_electrical_freq,
)
from src.thermal_network import (
    load_thermal_params, ThermalParams, ThermalState,
    simulate_thermal, N_THERMAL, IDX_MAGNET, NODE_NAMES,
)
from src.mpc_controller import (
    load_mpc_params, MPCParams, MPCController,
    coupled_dynamics, run_mpc_simulation,
    N_STATES, IDX_V, IDX_SOC, IDX_TH_START, IDX_TH_END,
)
from src.thermal_network import N_THERMAL, IDX_MAGNET


# %%
# ═══════════════════════════════════════════════════════════════════
# Stage 6: Plotting Suite
# ═══════════════════════════════════════════════════════════════════

def generate_all_plots(track: TrackData, results: dict,
                       motor: MotorParams, mpc_p: MPCParams,
                       plots_dir: Path):
    """Generate all Stage 6 output figures.

    Plots:
      1. Deployment profile (P_e vs distance with segment shading)
      2. T_magnet vs demagnetisation limit (the critical constraint)
      3. SOC trajectory
      4. All 5 thermal node temperatures
      5. Combined dashboard (4-panel overview)
      6. Loss breakdown over the lap
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    plots_dir.mkdir(parents=True, exist_ok=True)

    x_hist = results["x_hist"]
    u_hist = results["u_hist"]
    N = len(u_hist)

    t = track.t[:N]
    s = track.s[:N] / 1000  # km
    v_mpc = x_hist[:N, IDX_V] * 3.6  # km/h
    v_track = track.v[:N] * 3.6
    soc = x_hist[:N, IDX_SOC] * 100  # %
    T_all = x_hist[:N, IDX_TH_START:IDX_TH_END]
    P_e = u_hist[:N] / 1e3  # kW

    segment = track.segment[:N]
    is_corner = segment == "corner"

    # Helper: shade corners on an axis
    def shade_corners(ax, x_arr):
        corner_start = None
        for i in range(len(is_corner)):
            if is_corner[i] and corner_start is None:
                corner_start = i
            elif not is_corner[i] and corner_start is not None:
                ax.axvspan(x_arr[corner_start], x_arr[i - 1],
                           alpha=0.10, color="red", zorder=0)
                corner_start = None
        if corner_start is not None:
            ax.axvspan(x_arr[corner_start], x_arr[-1],
                       alpha=0.10, color="red", zorder=0)

    # ── Plot 1: Deployment Profile ────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("MGU-K Deployment Profile — Monza", fontsize=14, fontweight="bold")

    ax = axes[0]
    shade_corners(ax, s)
    ax.fill_between(s, 0, np.clip(P_e, 0, None), color="#DC0000",
                    alpha=0.6, label="Deploy")
    ax.fill_between(s, 0, np.clip(P_e, None, 0), color="#0050A0",
                    alpha=0.6, label="Regen")
    ax.axhline(mpc_p.P_peak / 1e3, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(-mpc_p.P_peak / 1e3, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylabel("P_e [kW]")
    ax.set_ylim(-400, 400)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    shade_corners(ax, s)
    ax.plot(s, v_track, color="#AAAAAA", linewidth=0.8, label="Track reference")
    ax.plot(s, v_mpc, color="#DC0000", linewidth=1.2, label="MPC (with MGU-K)")
    ax.set_ylabel("Speed [km/h]")
    ax.set_xlabel("Distance [km]")
    ax.set_ylim(0, 400)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / "deployment_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] deployment_profile.png")

    # ── Plot 2: T_magnet vs Demagnetisation Limit ─────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    T_mag = T_all[:, IDX_MAGNET]
    ax.plot(t, T_mag, color="#8B008B", linewidth=1.8, label="T_magnet (MPC)")
    ax.axhline(mpc_p.T_magnet_max, color="#DC0000", ls="--", linewidth=2.0,
               label=f"Demagnetisation limit ({mpc_p.T_magnet_max}°C)")
    ax.fill_between(t, T_mag, mpc_p.T_magnet_max,
                    where=(T_mag > mpc_p.T_magnet_max - 10),
                    alpha=0.2, color="red", label="Danger zone (<10°C margin)")
    shade_corners(ax, t)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Magnet Temperature vs Demagnetisation Limit", fontweight="bold")
    margin = mpc_p.T_magnet_max - T_mag.max()
    ax.text(0.98, 0.05, f"Min margin: {margin:.1f}°C",
            transform=ax.transAxes, fontsize=11, ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "T_magnet_vs_limit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] T_magnet_vs_limit.png")

    # ── Plot 3: SOC Trajectory ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, soc, color="#00A000", linewidth=1.5)
    ax.axhline(mpc_p.SOC_min * 100, color="red", ls="--", lw=1, label=f"SOC min ({mpc_p.SOC_min*100:.0f}%)")
    ax.axhline(mpc_p.SOC_max * 100, color="red", ls="--", lw=1, label=f"SOC max ({mpc_p.SOC_max*100:.0f}%)")
    ax.axhline(mpc_p.SOC_ref * 100, color="gray", ls=":", lw=1, label=f"SOC ref ({mpc_p.SOC_ref*100:.0f}%)")
    shade_corners(ax, t)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SOC [%]")
    ax.set_title("Battery State of Charge", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "soc_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] soc_trajectory.png")

    # ── Plot 4: All Thermal Nodes ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#DC0000", "#FF8C00", "#8B008B", "#0050A0", "#00A000"]
    labels = ["Winding", "Stator Iron", "Magnet", "Housing", "Coolant"]
    for i in range(N_THERMAL):
        ax.plot(t, T_all[:, i], color=colors[i], linewidth=1.3,
                label=f"{labels[i]} ({T_all[-1, i]:.1f}°C)")
    ax.axhline(mpc_p.T_magnet_max, color="#8B008B", ls="--", lw=1.5,
               alpha=0.7, label=f"Demag limit ({mpc_p.T_magnet_max}°C)")
    shade_corners(ax, t)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("5-Node Thermal Network — All Components", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "thermal_all_nodes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] thermal_all_nodes.png")

    # ── Plot 5: Combined Dashboard ────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[2, 1.5, 1, 1], hspace=0.3)
    fig.suptitle("MGU-K Thermal-Constrained ERS Deployment — Monza Full Lap",
                 fontsize=15, fontweight="bold", y=0.98)

    # Panel A: Speed
    ax0 = fig.add_subplot(gs[0])
    shade_corners(ax0, t)
    ax0.plot(t, v_track, color="#AAAAAA", lw=0.8, label="Track ref")
    ax0.plot(t, v_mpc, color="#DC0000", lw=1.2, label="MPC")
    ax0.set_ylabel("Speed [km/h]")
    ax0.set_ylim(0, 400)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_xticklabels([])

    # Panel B: Deployment
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    shade_corners(ax1, t)
    ax1.fill_between(t, 0, np.clip(P_e, 0, None), color="#DC0000", alpha=0.6)
    ax1.fill_between(t, 0, np.clip(P_e, None, 0), color="#0050A0", alpha=0.6)
    ax1.set_ylabel("P_e [kW]")
    ax1.set_ylim(-400, 400)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    # Panel C: T_magnet
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    shade_corners(ax2, t)
    ax2.plot(t, T_all[:, IDX_MAGNET], color="#8B008B", lw=1.5)
    ax2.axhline(mpc_p.T_magnet_max, color="red", ls="--", lw=1.5)
    ax2.set_ylabel("T_mag [°C]")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    # Panel D: SOC
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    shade_corners(ax3, t)
    ax3.plot(t, soc, color="#00A000", lw=1.3)
    ax3.axhline(mpc_p.SOC_min * 100, color="red", ls="--", lw=0.8)
    ax3.axhline(mpc_p.SOC_max * 100, color="red", ls="--", lw=0.8)
    ax3.set_ylabel("SOC [%]")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    fig.savefig(plots_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] dashboard.png")

    # ── Plot 6: Loss Breakdown ────────────────────────────────────
    P_cu, P_fe, P_tot = total_losses(u_hist[:N], x_hist[:N, IDX_V], motor)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(t, 0, P_cu / 1e3, color="#DC0000", alpha=0.5, label="Copper")
    ax.fill_between(t, P_cu / 1e3, (P_cu + P_fe) / 1e3, color="#FF8C00",
                    alpha=0.5, label="Iron")
    shade_corners(ax, t)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loss [kW]")
    ax.set_title("PMSM Loss Breakdown Over Lap", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "loss_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] loss_breakdown.png")


# %%
# ═══════════════════════════════════════════════════════════════════
# Multi-Lap Thermal Soak Analysis
# ═══════════════════════════════════════════════════════════════════

def run_multi_lap_simulation(track: "TrackData",
                              motor: "MotorParams", tp: "ThermalParams",
                              mpc_p: "MPCParams",
                              n_laps: int = 6,
                              SOC_init: float = 0.50,
                              verbose: bool = True) -> tuple:
    """Run n_laps consecutive MPC simulations carrying thermal state forward.

    Each lap begins where the previous one ended thermally — temperatures are
    NOT reset between laps (as in a real race stint). SOC is also carried
    forward so the controller must manage energy across the entire stint.

    Parameters
    ----------
    track    : TrackData (includes braking_zone for multi-rate horizon)
    n_laps   : number of consecutive laps to simulate
    SOC_init : starting SOC for lap 1

    Returns
    -------
    all_results   : list of result dicts (one per lap)
    lap_summaries : list of per-lap stat dicts
    """
    from src.thermal_network import load_thermal_params

    all_results = []
    lap_summaries = []

    # Thermal state: start cold (coolant inlet temperature)
    T_current = np.full(N_THERMAL, tp.T_coolant_inlet)
    SOC_current = SOC_init

    for lap in range(1, n_laps + 1):
        if verbose:
            print(f"\n  {'─' * 55}")
            print(f"  LAP {lap}/{n_laps}  |  "
                  f"T_magnet_init = {T_current[IDX_MAGNET]:.1f}°C  |  "
                  f"SOC_init = {SOC_current * 100:.1f}%")
            print(f"  {'─' * 55}")

        results = run_mpc_simulation(
            track.v, track.segment,
            motor, tp, mpc_p,
            T_init=T_current.copy(),
            SOC_init=SOC_current,
            track_braking_zone=track.braking_zone,
            verbose=verbose,
        )

        all_results.append(results)

        # ── Carry forward state to next lap ──
        x_final = results["x_hist"][-1]
        T_current = x_final[IDX_TH_START:IDX_TH_END].copy()
        SOC_current = float(x_final[IDX_SOC])

        # ── Lap summary statistics ──
        x_hist = results["x_hist"]
        u_hist = results["u_hist"]
        N = len(u_hist)
        T_mag = x_hist[:N, IDX_TH_START + IDX_MAGNET]
        soc   = x_hist[:N, IDX_SOC]
        P_e   = u_hist[:N]

        E_deploy = float(np.sum(np.clip(P_e, 0, None)) * mpc_p.dt)
        E_regen  = float(np.sum(np.clip(-P_e, 0, None)) * mpc_p.dt)

        summary = {
            "lap":        lap,
            "T_mag_init": float(x_hist[0, IDX_TH_START + IDX_MAGNET]),
            "T_mag_peak": float(T_mag.max()),
            "T_mag_end":  float(T_mag[-1]),
            "margin":     float(mpc_p.T_magnet_max - T_mag.max()),
            "SOC_start":  float(soc[0]),
            "SOC_end":    float(soc[-1]),
            "E_deploy_MJ": E_deploy / 1e6,
            "E_regen_MJ":  E_regen / 1e6,
        }
        lap_summaries.append(summary)

        status = ("OK" if summary["margin"] >= 5 else
                  "WARN" if summary["margin"] >= 1 else "CRITICAL")
        print(f"  → T_mag peak = {summary['T_mag_peak']:.1f}°C  "
              f"(margin = {summary['margin']:.1f}°C  [{status}])  |  "
              f"SOC {summary['SOC_start']*100:.1f}% → {summary['SOC_end']*100:.1f}%")

    return all_results, lap_summaries


def generate_multi_lap_plots(all_results: list, lap_summaries: list,
                             track: "TrackData", mpc_p: "MPCParams",
                             plots_dir: "Path"):
    """Generate all multi-lap output figures.

    Produces six plot groups (10-lap aggregate + last-lap):
      1. Heat & Loss Dashboard  — all 5 thermal nodes + stacked losses vs distance
      2. Summary Dashboard      — MPC speed + T_magnet vs distance
      3. Loss Breakdown         — copper / iron stacked area vs distance
      4. Efficiency Map         — analytical contourf heatmap (speed × power)
      5. Losses vs Speed        — analytical line plots at fixed power levels
      6. Loss Maps vs Power     — analytical line plots at fixed speeds
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from src.thermal_network import N_THERMAL, IDX_MAGNET
    from src.mpc_controller import IDX_SOC, IDX_V
    from src.pmsm_losses import (total_losses, load_motor_params,
                                 plot_efficiency_map, plot_loss_vs_speed,
                                 plot_loss_map, _analytical_losses)

    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Concatenate lap data into continuous arrays ────────────────────
    t_all, s_all = [], []
    SOC_all, v_all, P_all, T_nodes_all = [], [], [], []

    t_offset = 0.0
    s_offset = 0.0
    for res in all_results:
        N = len(res["u_hist"])
        t_all.append(track.t[:N] + t_offset)
        s_all.append((track.s[:N] + s_offset) / 1000.0)   # km
        SOC_all.append(res["x_hist"][:N, IDX_SOC] * 100)
        v_all.append(res["x_hist"][:N, IDX_V] * 3.6)      # km/h
        P_all.append(res["u_hist"][:N] / 1e3)              # kW
        T_nodes_all.append(res["x_hist"][:N, 2:2 + N_THERMAL])
        t_offset += track.lap_time
        s_offset += track.s[-1]

    t_all      = np.concatenate(t_all)
    s_all      = np.concatenate(s_all)
    SOC_all    = np.concatenate(SOC_all)
    v_all      = np.concatenate(v_all)
    P_all      = np.concatenate(P_all)
    T_nodes_all = np.concatenate(T_nodes_all)
    T_mag_all  = T_nodes_all[:, IDX_MAGNET]

    # Loss computation over all laps
    motor    = load_motor_params()
    P_e_W    = P_all * 1e3
    v_ms     = v_all / 3.6
    P_cu, P_fe, P_tot = total_losses(P_e_W, v_ms, motor)

    # Last-lap slice (for single-lap detail plots)
    last_N     = len(all_results[-1]["u_hist"])
    s_last     = s_all[-last_N:] - s_all[-last_N]
    v_last     = v_all[-last_N:]
    P_last     = P_all[-last_N:]
    T_nodes_last = T_nodes_all[-last_N:]
    T_mag_last = T_mag_all[-last_N:]
    P_cu_last  = P_cu[-last_N:]
    P_fe_last  = P_fe[-last_N:]
    P_tot_last = P_tot[-last_N:]
    t_last     = t_all[-last_N:] - t_all[-last_N]  # reset to 0

    # ── Node display settings ──────────────────────────────────────────
    NODE_COLORS  = ["#DC0000", "#FF8C00", "#8B008B", "#0050A0", "#00A000"]
    NODE_LABELS  = ["Winding Temp", "Stator Iron Temp", "Magnet Temp",
                    "Housing Temp", "Coolant Temp"]

    # ─────────────────────────────────────────────────────────────────
    # Helper: Plot 1 — Heat & Loss Dashboard (key target graph)
    # Top:    all 5 thermal node temperatures vs distance [km]
    # Bottom: stacked copper + iron losses vs distance [km]
    # ─────────────────────────────────────────────────────────────────
    def make_heat_loss_dashboard(prefix, s, T_nodes, P_c, P_f, t_label):
        n_laps_str = "10 Laps" if "10" in prefix else "Last Lap"
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(
            f"MGU-K Component Heat & Loss Generation [{n_laps_str}]",
            fontsize=14, fontweight="bold")

        # Top panel: temperatures
        ax0 = axes[0]
        for i in range(N_THERMAL):
            ax0.plot(s, T_nodes[:, i], color=NODE_COLORS[i],
                     linewidth=1.3, label=NODE_LABELS[i])
        ax0.set_ylabel("Node Temperatures [°C]", fontsize=11)
        ax0.legend(loc="upper right", fontsize=9)
        ax0.grid(True, alpha=0.3)
        ax0.set_ylim(bottom=60)

        # Bottom panel: stacked losses
        ax1 = axes[1]
        ax1.fill_between(s, 0, P_c / 1e3, color="#DC0000",
                         alpha=0.75, label="Copper Losses (Winding)")
        ax1.fill_between(s, P_c / 1e3, (P_c + P_f) / 1e3, color="#FF8C00",
                         alpha=0.75, label="Iron Losses (Stator)")
        ax1.set_ylabel("Power Loss [kW]", fontsize=11)
        ax1.set_xlabel("Continuous Distance [km]", fontsize=11)
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        plt.tight_layout()
        fname = plots_dir / f"{prefix}_heat_loss_dashboard.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] {fname.name}")

    # ─────────────────────────────────────────────────────────────────
    # Helper: Plot 2 — Summary Dashboard (speed + T_magnet)
    # ─────────────────────────────────────────────────────────────────
    def make_summary_dashboard(prefix, s, v, P, T_mag):
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        n_str = "10 Laps" if "10" in prefix else "Last Lap"
        fig.suptitle(f"MGU-K MPC Summary — Monza [{n_str}]",
                     fontsize=14, fontweight="bold")

        ax0 = axes[0]
        ax0.plot(s, v, color="#DC0000", lw=1.2, label="MPC Speed")
        ax0.fill_between(s, 0, 400, where=(P < -10),
                         color="blue", alpha=0.10, label="Braking/Regen")
        ax0.fill_between(s, 0, 400, where=(P > 10),
                         color="red",  alpha=0.10, label="Deployment")
        ax0.set_ylabel("Speed [km/h]")
        ax0.set_ylim(0, 400)
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1]
        ax1.plot(s, T_mag, color="#8B008B", lw=1.5, label="T_magnet (MPC)")
        ax1.axhline(mpc_p.T_magnet_max, color="red", ls="--", lw=2.0,
                    label=f"Demag limit ({mpc_p.T_magnet_max}°C)")
        ax1.axhline(mpc_p.T_magnet_max - 10, color="#FF8C00", ls="--",
                    lw=1.5, label="Danger zone (−10°C)")
        ax1.fill_between(s, mpc_p.T_magnet_max - 10, mpc_p.T_magnet_max,
                         alpha=0.10, color="red")
        ax1.set_ylabel("Temperature [°C]")
        ax1.set_xlabel("Continuous Distance [km]")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = plots_dir / f"{prefix}_summary_dashboard.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] {fname.name}")

    # ─────────────────────────────────────────────────────────────────
    # Helper: Plot 3 — Loss Breakdown (stacked area, for reference)
    # ─────────────────────────────────────────────────────────────────
    def make_loss_breakdown(prefix, s, P_c, P_f):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(s, 0, P_c / 1e3, color="#DC0000",
                        alpha=0.55, label="Copper Losses (Winding)")
        ax.fill_between(s, P_c / 1e3, (P_c + P_f) / 1e3, color="#FF8C00",
                        alpha=0.55, label="Iron Losses (Stator)")
        n_str = "10 Laps" if "10" in prefix else "Last Lap"
        ax.set_title(f"PMSM Loss Breakdown — {n_str}", fontweight="bold")
        ax.set_ylabel("Power Loss [kW]")
        ax.set_xlabel("Continuous Distance [km]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fname = plots_dir / f"{prefix}_loss_breakdown.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] {fname.name}")

    # ─────────────────────────────────────────────────────────────────
    # Helper: Plots 4–6 — Analytical loss & efficiency maps
    # These do NOT depend on simulation data — they are computed from
    # the PMSM model across the full operating envelope.
    # ─────────────────────────────────────────────────────────────────
    def make_analytical_maps(prefix):
        # MGU-K Efficiency Map (contourf heatmap)
        plot_efficiency_map(
            motor,
            save_path=str(plots_dir / f"{prefix}_pmsm_efficiency_map.png"),
            show=False)
        print(f"  [plot] {prefix}_pmsm_efficiency_map.png")

        # PMSM Losses vs Vehicle Speed (line plots at fixed power levels)
        plot_loss_vs_speed(
            motor,
            save_path=str(plots_dir / f"{prefix}_pmsm_losses_vs_speed.png"),
            show=False)
        print(f"  [plot] {prefix}_pmsm_losses_vs_speed.png")

        # PMSM Loss Maps vs Electrical Power (line plots at fixed speeds)
        plot_loss_map(
            motor,
            save_path=str(plots_dir / f"{prefix}_pmsm_losses_vs_power.png"),
            show=False)
        print(f"  [plot] {prefix}_pmsm_losses_vs_power.png")

    # ── Generate 10-lap plots ─────────────────────────────────────────
    print("\n  [plots] 10-lap aggregate figures …")
    make_heat_loss_dashboard("10_lap", s_all, T_nodes_all, P_cu, P_fe, t_all)
    make_summary_dashboard("10_lap", s_all, v_all, P_all, T_mag_all)
    make_loss_breakdown("10_lap", s_all, P_cu, P_fe)
    make_analytical_maps("10_lap")

    # ── Generate last-lap plots ───────────────────────────────────────
    print("\n  [plots] Last-lap detail figures …")
    make_heat_loss_dashboard("last_lap", s_last, T_nodes_last,
                             P_cu_last, P_fe_last, t_last)
    make_summary_dashboard("last_lap", s_last, v_last, P_last, T_mag_last)
    make_loss_breakdown("last_lap", s_last, P_cu_last, P_fe_last)
    make_analytical_maps("last_lap")

def simulate_and_plot_fixed_strategy(track: "TrackData",
                                     motor: "MotorParams",
                                     tp: "ThermalParams",
                                     plots_dir: Path,
                                     P_deploy_kW: float = 200.0,
                                     P_regen_kW: float = 80.0):
    """Simulate 1 lap with a FIXED deployment strategy (no MPC) and plot
    all 5 thermal node temperatures vs time.

    This is the motivating plot: it demonstrates that a naive fixed
    strategy drives temperatures dangerously close to (or past) the
    140°C demagnetisation limit — exactly why coupled thermal-electrical
    MPC is required.

    Strategy:
      - Straight segments  → deploy  +P_deploy_kW kW
      - Corner  segments   → regen   −P_regen_kW  kW (braking energy recovery)

    Parameters
    ----------
    track       : TrackData  — Monza speed and segment profile
    motor       : MotorParams
    tp          : ThermalParams
    plots_dir   : Path  — where to save the figure
    P_deploy_kW : peak deployment power [kW]
    P_regen_kW  : peak regen power [kW] (sign convention: positive value)
    """
    import matplotlib.pyplot as plt
    from src.thermal_network import ThermalState, step_euler, N_THERMAL, IDX_MAGNET
    from src.pmsm_losses import _analytical_losses as _losses

    N  = len(track.v)
    dt = track.dt

    # Start cold — all nodes at coolant inlet temperature
    state  = ThermalState(T=np.full(N_THERMAL, tp.T_coolant_inlet))
    T_hist = np.zeros((N, N_THERMAL))

    for k in range(N):
        v_k  = track.v[k]
        seg  = track.segment[k]
        P_e  = P_deploy_kW * 1e3 if seg == "straight" else -P_regen_kW * 1e3

        # Analytical losses (fixed B_pk=1.5), no speed-enhanced cooling —
        # conservative baseline to reveal worst-case naive strategy risk.
        P_cu_k, P_fe_k, _ = _losses(
            np.array([P_e]), np.array([v_k]), motor)
        state = step_euler(state, float(P_cu_k[0]), float(P_fe_k[0]),
                           dt, tp, v_vehicle=0.0)
        T_hist[k] = state.T

    # ── Plot ──────────────────────────────────────────────────────────
    NODE_COLORS = ["#DC0000", "#FF8C00", "#8B008B", "#0050A0", "#00A000"]
    NODE_LABELS_FULL = ["Winding", "Stator Iron", "Magnet", "Housing", "Coolant"]

    fig, ax = plt.subplots(figsize=(13, 6))
    t = track.t[:N]

    for i in range(N_THERMAL):
        lbl = f"{NODE_LABELS_FULL[i]} ({T_hist[-1, i]:.1f}°C)"
        ax.plot(t, T_hist[:, i], color=NODE_COLORS[i], linewidth=2.0, label=lbl)

    ax.axhline(140.0, color="#8B008B", ls="--", linewidth=2.0,
               label="Demag limit (140°C)")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.set_title(
        f"Monza Lap: {P_deploy_kW:.0f} kW deploy / {P_regen_kW:.0f} kW regen",
        fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t[-1])

    plt.tight_layout()
    fname = plots_dir / "fixed_strategy_thermal.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {fname.name}")

    T_mag_peak = T_hist[:, IDX_MAGNET].max()
    print(f"  Fixed strategy: T_magnet peak = {T_mag_peak:.1f}°C  "
          f"({'OK' if T_mag_peak < 140 else 'EXCEEDS LIMIT!'})")
    return T_hist


def print_multi_lap_summary(lap_summaries: list, mpc_p: "MPCParams"):
    """Print a formatted table of per-lap results."""
    print(f"\n{'═' * 70}")
    print(f"  MULTI-LAP THERMAL SOAK SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Lap':>4}  {'T_init':>8}  {'T_peak':>8}  {'Margin':>8}  "
          f"{'SOC_in':>7}  {'SOC_out':>7}  {'Status':>8}")
    print(f"  {'─' * 66}")
    for s in lap_summaries:
        status = ("  OK    " if s["margin"] >= 5 else
                  "  WARN  " if s["margin"] >= 1 else " CRITICAL")
        print(f"  {s['lap']:>4}  "
              f"{s['T_mag_init']:>7.1f}°C  "
              f"{s['T_mag_peak']:>7.1f}°C  "
              f"{s['margin']:>7.1f}°C  "
              f"{s['SOC_start']*100:>6.1f}%  "
              f"{s['SOC_end']*100:>6.1f}%  "
              f"{status}")
    print(f"{'═' * 70}")
    final = lap_summaries[-1]
    print(f"\n  KEY FINDING: After {len(lap_summaries)} laps of thermal soak,")
    print(f"  T_magnet peaks at {final['T_mag_peak']:.1f}°C "
          f"({final['margin']:.1f}°C margin vs {mpc_p.T_magnet_max}°C limit).")
    if final["margin"] >= 5:
        print(f"  Thermal equilibrium is STABLE — controller maintains safety margin.")
    elif final["margin"] >= 1:
        print(f"  Thermal margin is TIGHT — consider tightening barrier weight.")
    else:
        print(f"  WARNING: Thermal margin critically low — controller needs retuning.")
    print(f"{'═' * 70}")


# %%
# ═══════════════════════════════════════════════════════════════════
# Summary Statistics
# ═══════════════════════════════════════════════════════════════════

def print_summary(track: TrackData, results: dict, mpc_p: MPCParams,
                  motor: MotorParams, elapsed: float):
    """Print a comprehensive summary of the simulation results."""
    x_hist = results["x_hist"]
    u_hist = results["u_hist"]
    N = len(u_hist)

    v_mpc = x_hist[:N, IDX_V]
    soc = x_hist[:N, IDX_SOC]
    T_mag = x_hist[:N, IDX_TH_START + IDX_MAGNET]
    P_e = u_hist[:N]

    # Energy accounting
    E_deploy = np.sum(np.clip(P_e, 0, None)) * mpc_p.dt
    E_regen  = np.sum(np.clip(-P_e, 0, None)) * mpc_p.dt
    E_net    = np.sum(P_e) * mpc_p.dt

    # Losses
    P_cu, P_fe, P_tot = total_losses(P_e, v_mpc, motor)
    E_loss = np.sum(P_tot) * mpc_p.dt

    # Constraint margins
    T_mag_peak = T_mag.max()
    T_mag_margin = mpc_p.T_magnet_max - T_mag_peak
    soc_min_actual = soc.min()
    soc_max_actual = soc.max()

    # Solver stats
    info_list = results["info_hist"]
    statuses = [i.get("status", "?") for i in info_list]
    solved_count = sum(1 for s in statuses if "solved" in s.lower())

    print(f"\n{'═' * 60}")
    print(f"  SIMULATION RESULTS SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Track       : Monza ({track.source})")
    print(f"  Lap time    : {track.lap_time:.2f} s")
    print(f"  MPC steps   : {N}")
    print(f"  Wall time   : {elapsed:.1f} s ({elapsed/N*1000:.1f} ms/step)")
    print(f"{'─' * 60}")
    print(f"  DEPLOYMENT")
    print(f"    P_e mean    : {P_e.mean()/1e3:+.1f} kW")
    print(f"    P_e max     : {P_e.max()/1e3:+.1f} kW (deploy)")
    print(f"    P_e min     : {P_e.min()/1e3:+.1f} kW (regen)")
    print(f"    E_deploy    : {E_deploy/1e6:.2f} MJ")
    print(f"    E_regen     : {E_regen/1e6:.2f} MJ")
    print(f"    E_net       : {E_net/1e6:.2f} MJ")
    print(f"    E_net < 4 MJ? {'YES' if abs(E_net) < 4e6 else 'NO'}")
    print(f"{'─' * 60}")
    print(f"  THERMAL")
    print(f"    T_magnet peak  : {T_mag_peak:.1f} °C")
    print(f"    T_magnet margin: {T_mag_margin:.1f} °C to demag limit")
    print(f"    T_magnet < 140°C? {'YES' if T_mag_peak < mpc_p.T_magnet_max else 'NO — VIOLATION'}")
    print(f"    E_loss total   : {E_loss/1e3:.1f} kJ")
    print(f"    Mean P_loss    : {P_tot.mean()/1e3:.2f} kW")
    print(f"{'─' * 60}")
    print(f"  SOC")
    print(f"    SOC start   : {soc[0]*100:.1f}%")
    print(f"    SOC end     : {soc[-1]*100:.1f}%")
    print(f"    SOC min     : {soc_min_actual*100:.1f}%")
    print(f"    SOC max     : {soc_max_actual*100:.1f}%")
    print(f"{'─' * 60}")
    print(f"  SPEED")
    print(f"    v_mean (MPC)  : {v_mpc.mean()*3.6:.1f} km/h")
    print(f"    v_mean (track): {track.v[:N].mean()*3.6:.1f} km/h")
    print(f"    Δv_mean       : {(v_mpc.mean() - track.v[:N].mean())*3.6:+.1f} km/h")
    print(f"{'─' * 60}")
    print(f"  SOLVER")
    print(f"    Solved        : {solved_count}/{N} ({100*solved_count/N:.0f}%)")
    print(f"{'═' * 60}")


# %%
# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MGU-K Thermal-Constrained ERS Deployment Optimiser")
    print("=" * 60)

    from src.track_model import load_track, load_params
    from src.pmsm_losses import load_motor_params
    from src.thermal_network import load_thermal_params
    from src.mpc_controller import load_mpc_params

    params = load_params()
    motor  = load_motor_params()
    tp     = load_thermal_params()
    mpc_p  = load_mpc_params()

    track = load_track(params)
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fixed-strategy motivating plot ─────────────────────
    # Shows why MPC is needed: naive 200 kW / 80 kW deployment drives
    # temperatures dangerously close to the demagnetisation limit.
    print("\n[1/4] Simulating fixed-strategy lap (200 kW deploy / 80 kW regen)...")
    simulate_and_plot_fixed_strategy(track, motor, tp, plots_dir,
                                     P_deploy_kW=200.0, P_regen_kW=80.0)

    # ── Step 2: 10-lap MPC simulation ─────────────────────────────
    # SOC oscillates between 80% and 20% across the stint.
    mpc_p.SOC_max = 0.80
    mpc_p.SOC_min = 0.20
    mpc_p.SOC_ref = 0.65
    mpc_p.w_soc   = 250.0   # stiff spring back to SOC_ref between laps

    print("\n[2/4] Running 10-lap MPC simulation (thermal soak analysis)...")
    all_lap_results, lap_summaries = run_multi_lap_simulation(
        track, motor, tp, mpc_p,
        n_laps=10,
        SOC_init=0.80,
        verbose=False,
    )
    print_multi_lap_summary(lap_summaries, mpc_p)

    # ── Step 3: Generate all multi-lap and last-lap plots ──────────
    print("\n[3/4] Generating 10-lap & last-lap plots...")
    generate_multi_lap_plots(all_lap_results, lap_summaries,
                             track, mpc_p, plots_dir)

    print(f"\n{'=' * 60}")
    print(f"  All plots saved to {plots_dir}/")
    print(f"{'=' * 60}")
    print("\n✅ Simulation complete!")

if __name__ == "__main__":
    main()
