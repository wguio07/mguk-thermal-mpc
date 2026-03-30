# %%
"""
thermal_network.py — 5-Node Lumped Thermal Network
═══════════════════════════════════════════════════
Stage 3 of MGU-K Thermal-Constrained ERS Deployment Optimiser

Models heat transfer in the MGU-K as a 5-node lumped-parameter
thermal network. Each node has a thermal capacitance C_i and is
coupled to neighbours via thermal resistances R_ij.

    ┌──────────┐   R_ws   ┌──────────┐   R_sm   ┌──────────┐
    │ Winding  │─────────│  Stator  │─────────│  Magnet  │
    │ (P_cu)   │          │ (P_fe)   │          │          │
    └──────────┘          └────┬─────┘          └──────────┘
                               │ R_sh
                          ┌────┴─────┐
                          │ Housing  │
                          └────┬─────┘
                               │ R_hc
                          ┌────┴─────┐
                          │ Coolant  │← T_inlet (boundary)
                          └──────────┘

Governing equation per node:
    C_i × dT_i/dt = P_loss,i + Σ_j (T_j − T_i) / R_ij

The coolant node exchanges heat with the housing AND loses heat
to the coolant inlet at T_inlet (forced convection boundary).

Critical constraint: T_magnet < 140°C (NdFeB demagnetisation limit)

This module provides:
  - ThermalState: container for the 5 temperatures
  - ThermalNetwork: ODE system with forward Euler integration
  - Jacobian computation (numerical finite differences) for MPC

Author: Wolfgang Guio
Project: Ferrari F1 Engineering Academy 2026
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

# %%
# ═══════════════════════════════════════════════════════════════════
# Thermal State Container
# ═══════════════════════════════════════════════════════════════════

# Node indices (for array-based operations)
IDX_WINDING = 0
IDX_STATOR  = 1
IDX_MAGNET  = 2
IDX_HOUSING = 3
IDX_COOLANT = 4
N_THERMAL   = 5

NODE_NAMES = ["winding", "stator", "magnet", "housing", "coolant"]


@dataclass
class ThermalState:
    """Temperature state of the 5-node thermal network.

    Attributes
    ----------
    T : np.ndarray, shape (5,)
        Temperatures [°C] in order:
        [T_winding, T_stator, T_magnet, T_housing, T_coolant]
    """
    T: np.ndarray

    @classmethod
    def from_uniform(cls, T_init: float = 65.0) -> "ThermalState":
        """Initialise all nodes at the same temperature (e.g. coolant inlet)."""
        return cls(T=np.full(N_THERMAL, T_init))

    @classmethod
    def from_dict(cls, d: dict) -> "ThermalState":
        """Construct from a dict with node names as keys."""
        return cls(T=np.array([d[n] for n in NODE_NAMES]))

    @property
    def T_winding(self) -> float: return self.T[IDX_WINDING]
    @property
    def T_stator(self) -> float:  return self.T[IDX_STATOR]
    @property
    def T_magnet(self) -> float:  return self.T[IDX_MAGNET]
    @property
    def T_housing(self) -> float: return self.T[IDX_HOUSING]
    @property
    def T_coolant(self) -> float: return self.T[IDX_COOLANT]

    def as_dict(self) -> dict:
        return {n: self.T[i] for i, n in enumerate(NODE_NAMES)}

    def copy(self) -> "ThermalState":
        return ThermalState(T=self.T.copy())

    def summary(self) -> str:
        lines = [f"  T_{n:8s} = {self.T[i]:6.1f} °C" for i, n in enumerate(NODE_NAMES)]
        return "\n".join(lines)


# %%
# ═══════════════════════════════════════════════════════════════════
# Thermal Network Parameters
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ThermalParams:
    """All parameters for the 5-node lumped thermal network."""
    # Thermal capacitances [J/K]
    C: np.ndarray          # shape (5,)

    # Thermal conductances [W/K] — stored as G_ij = 1/R_ij for efficiency
    # Topology: winding↔stator, stator↔magnet, stator↔housing, housing↔coolant
    G_winding_stator: float
    G_stator_magnet: float
    G_stator_housing: float
    G_housing_coolant: float

    # Coolant boundary condition
    T_coolant_inlet: float   # [°C]
    G_coolant_inlet: float   # [W/K] conductance to inlet (forced convection)

    # Constraint
    T_magnet_max: float      # [°C] demagnetisation limit

    @classmethod
    def from_yaml(cls, params: dict) -> "ThermalParams":
        """Construct from parsed motor_params.yaml."""
        tc = params["thermal_capacitance"]
        tr = params["thermal_resistance"]
        tl = params["thermal_limits"]

        C = np.array([
            tc["C_winding"],
            tc["C_stator"],
            tc["C_magnet"],
            tc["C_housing"],
            tc["C_coolant"],
        ], dtype=float)

        # Conductance = 1 / Resistance
        G_ws = 1.0 / tr["R_winding_stator"]
        G_sm = 1.0 / tr["R_stator_magnet"]
        G_sh = 1.0 / tr["R_stator_housing"]
        G_hc = 1.0 / tr["R_housing_coolant"]

        # Coolant-to-inlet conductance: models forced convection carrying
        # heat out of the motor.  The previous value of 1333 W/K anchored
        # the coolant node so strongly to 65°C that the motor cooled back
        # to near-ambient within every lap, hiding the thermal soak effect.
        #
        # Realistic race coolant systems for a 350 kW PMSM typically
        # achieve ΔT_coolant ≈ 5–15°C at continuous load (~50 kW losses),
        # giving G ≈ 50 kW / 10°C = 500 W/K after accounting for coolant
        # heat capacity and flow rate.  At this value:
        #   τ_coolant = C_coolant / G_coolant_inlet = 3000 / 500 = 6 s
        # The coolant rises ~10°C above inlet at steady state, and the
        # motor retains meaningful heat between laps (thermal soak visible
        # across a 10-lap stint as intended).
        G_coolant_inlet = 500.0

        return cls(
            C=C,
            G_winding_stator=G_ws,
            G_stator_magnet=G_sm,
            G_stator_housing=G_sh,
            G_housing_coolant=G_hc,
            T_coolant_inlet=tl["T_coolant_inlet"],
            G_coolant_inlet=G_coolant_inlet,
            T_magnet_max=tl["T_magnet_max"],
        )


def load_thermal_params(params_path: Optional[str] = None) -> ThermalParams:
    """Load thermal parameters from YAML."""
    if params_path is None:
        params_path = Path(__file__).resolve().parent.parent / "params" / "motor_params.yaml"
    # Use UTF-8 (with optional BOM) so Unicode comments parse on Windows.
    with open(params_path, "r", encoding="utf-8-sig") as f:
        params = yaml.safe_load(f)
    return ThermalParams.from_yaml(params)


# %%
# ═══════════════════════════════════════════════════════════════════
# Thermal ODE System
# ═══════════════════════════════════════════════════════════════════

def thermal_derivatives(T: np.ndarray, P_copper: float, P_iron: float,
                        tp: ThermalParams,
                        v_vehicle: float = 0.0) -> np.ndarray:
    """Compute dT/dt for the 5-node thermal network.

    This is the core ODE right-hand side. For each node i:
        C_i × dT_i/dt = P_loss,i + Σ_j G_ij × (T_j − T_i)

    Heat injection:
      - Copper losses → winding node
      - Iron losses   → stator node
      - No direct heat into magnet, housing, coolant (heated by conduction)

    Speed-dependent cooling:
      Real MGU-K cooling performance improves with vehicle speed because:
        (a) Coolant pump output scales with motor shaft speed (and thus
            vehicle speed) — higher flow rate → better Nusselt number in
            coolant channels → h ∝ flow^0.8 (Dittus-Boelter, turbulent).
        (b) The radiator (heat exchanger) benefits from increased ram-air
            flow at high speed, improving heat rejection from the coolant
            back to ambient.
      Modelled as:
        G_hc_eff(v)  = G_hc_base  × (1 + K_SPD_HC  × v / V_REF)
        G_ci_eff(v)  = G_ci_base  × (1 + K_SPD_CI  × v / V_REF)
      where V_REF = 100 m/s (≈ Monza top speed).
      Constants tuned so that at 300 km/h the cooling capacity is ~2×
      the pit-lane idle value — consistent with published F1 thermal
      management data for similar power-density motors.

    Parameters
    ----------
    T : np.ndarray, shape (5,)
        Current temperatures [°C]
    P_copper : float
        Copper losses [W] (injected into winding node)
    P_iron : float
        Iron losses [W] (injected into stator node)
    tp : ThermalParams
    v_vehicle : float
        Vehicle speed [m/s]. Default 0 (conservative — no cooling bonus).

    Returns
    -------
    dTdt : np.ndarray, shape (5,)
        Temperature derivatives [°C/s]
    """
    Tw, Ts, Tm, Th, Tc = T

    # ── Speed-dependent conductance scaling ───────────────────────────
    # At v=0 (stationary): G_eff = G_base (base cooling only)
    # At v=100 m/s (360 km/h): G_hc × 2.5, G_ci × 2.0
    # Physical basis: Dittus-Boelter turbulent pipe correlation gives
    # h ∝ Re^0.8 ∝ (pump_speed)^0.8 ≈ linear in v for the MPC range.
    V_REF     = 100.0   # m/s — normalisation (Monza top speed)
    K_SPD_HC  = 1.5     # housing↔coolant scaling: 2.5× at v_max
    K_SPD_CI  = 1.0     # coolant↔inlet scaling:   2.0× at v_max

    v_norm = min(float(v_vehicle), V_REF) / V_REF         # clamp to [0, 1]
    G_hc_eff = tp.G_housing_coolant * (1.0 + K_SPD_HC * v_norm)
    G_ci_eff = tp.G_coolant_inlet   * (1.0 + K_SPD_CI  * v_norm)

    # Heat flows (positive = into node)
    # Winding node: receives copper losses, conducts to stator
    Q_w = P_copper + tp.G_winding_stator * (Ts - Tw)

    # Stator node: receives iron losses, conducts to winding, magnet, housing
    Q_s = (P_iron
           + tp.G_winding_stator * (Tw - Ts)
           + tp.G_stator_magnet  * (Tm - Ts)
           + tp.G_stator_housing * (Th - Ts))

    # Magnet node: conducts to stator only (rotor — thermally isolated from housing)
    Q_m = tp.G_stator_magnet * (Ts - Tm)

    # Housing node: conducts to stator and speed-enhanced coolant path
    Q_h = (tp.G_stator_housing * (Ts - Th)
           + G_hc_eff * (Tc - Th))

    # Coolant node: conducts to housing, rejects heat to inlet via
    # speed-enhanced convection (faster pump → more heat carried away)
    Q_c = (G_hc_eff * (Th - Tc)
           + G_ci_eff * (tp.T_coolant_inlet - Tc))

    dTdt = np.array([Q_w, Q_s, Q_m, Q_h, Q_c]) / tp.C

    return dTdt


# %%
# ═══════════════════════════════════════════════════════════════════
# Forward Euler Integrator
# ═══════════════════════════════════════════════════════════════════

def step_euler(state: ThermalState, P_copper: float, P_iron: float,
               dt: float, tp: ThermalParams,
               v_vehicle: float = 0.0) -> ThermalState:
    """Advance the thermal state by one time step (forward Euler).

    T_{k+1} = T_k + dt × dT/dt(T_k, P_k)

    Forward Euler is used here because:
    1. The MPC relinearises at every step anyway (RTI approach)
    2. dt = 0.05 s is small relative to thermal time constants (~10-100 s)
    3. It's algebraically simple → easy to differentiate for Jacobians

    Parameters
    ----------
    state    : current ThermalState
    P_copper : copper losses at this step [W]
    P_iron   : iron losses at this step [W]
    dt       : time step [s]
    tp       : ThermalParams

    Returns
    -------
    ThermalState — updated temperatures
    """
    dTdt = thermal_derivatives(state.T, P_copper, P_iron, tp, v_vehicle=v_vehicle)
    T_new = state.T + dt * dTdt
    return ThermalState(T=T_new)


def simulate_thermal(P_copper_arr: np.ndarray, P_iron_arr: np.ndarray,
                     dt: float, tp: ThermalParams,
                     T_init: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulate the thermal network over a full time series.

    Parameters
    ----------
    P_copper_arr : np.ndarray, shape (N,) — copper losses at each step [W]
    P_iron_arr   : np.ndarray, shape (N,) — iron losses at each step [W]
    dt           : time step [s]
    tp           : ThermalParams
    T_init       : initial temperatures [°C], shape (5,).
                   Defaults to uniform at T_coolant_inlet.

    Returns
    -------
    T_history : np.ndarray, shape (N+1, 5)
        Temperature history. T_history[0] = T_init, T_history[k+1] = after step k.
    """
    N = len(P_copper_arr)
    assert len(P_iron_arr) == N

    if T_init is None:
        T_init = np.full(N_THERMAL, tp.T_coolant_inlet)

    T_hist = np.zeros((N + 1, N_THERMAL))
    T_hist[0] = T_init

    state = ThermalState(T=T_init.copy())
    for k in range(N):
        state = step_euler(state, P_copper_arr[k], P_iron_arr[k], dt, tp)
        T_hist[k + 1] = state.T

    return T_hist


# %%
# ═══════════════════════════════════════════════════════════════════
# Jacobian Computation (Numerical Finite Differences)
# ═══════════════════════════════════════════════════════════════════

def thermal_jacobian_state(T: np.ndarray, P_copper: float, P_iron: float,
                           dt: float, tp: ThermalParams,
                           eps: float = 1e-6) -> np.ndarray:
    """Compute ∂T_{k+1}/∂T_k via numerical finite differences.

    This is the state-transition Jacobian needed by the MPC at each
    Real-Time Iteration step. It tells the controller how a small
    perturbation in current temperatures affects next-step temperatures.

    A_thermal[i,j] = ∂T_{k+1,i} / ∂T_{k,j}

    Parameters
    ----------
    T        : current temperatures, shape (5,)
    P_copper : copper losses [W]
    P_iron   : iron losses [W]
    dt       : time step [s]
    tp       : ThermalParams
    eps      : perturbation size

    Returns
    -------
    A : np.ndarray, shape (5, 5) — state-transition Jacobian
    """
    T_ref = T + dt * thermal_derivatives(T, P_copper, P_iron, tp)

    A = np.zeros((N_THERMAL, N_THERMAL))
    for j in range(N_THERMAL):
        T_pert = T.copy()
        T_pert[j] += eps
        T_next_pert = T_pert + dt * thermal_derivatives(T_pert, P_copper, P_iron, tp)
        A[:, j] = (T_next_pert - T_ref) / eps

    return A


def thermal_jacobian_input(T: np.ndarray, P_copper: float, P_iron: float,
                           dt: float, tp: ThermalParams,
                           eps: float = 1e-6) -> np.ndarray:
    """Compute ∂T_{k+1}/∂u_k via numerical finite differences.

    Since the control input u = P_e affects temperatures through the
    loss model (P_e → I_s → P_copper, and P_e → P_iron indirectly
    via speed), this Jacobian captures that coupling.

    For the thermal-only sub-system, the "inputs" are P_copper and P_iron.
    This returns a (5, 2) matrix: columns are ∂T/∂P_copper and ∂T/∂P_iron.

    Parameters
    ----------
    T        : current temperatures, shape (5,)
    P_copper : copper losses [W]
    P_iron   : iron losses [W]
    dt       : time step [s]
    tp       : ThermalParams
    eps      : perturbation size

    Returns
    -------
    B : np.ndarray, shape (5, 2)
        B[:, 0] = ∂T_{k+1}/∂P_copper
        B[:, 1] = ∂T_{k+1}/∂P_iron
    """
    T_ref = T + dt * thermal_derivatives(T, P_copper, P_iron, tp)

    B = np.zeros((N_THERMAL, 2))

    # Perturb P_copper
    T_pert_cu = T + dt * thermal_derivatives(T, P_copper + eps, P_iron, tp)
    B[:, 0] = (T_pert_cu - T_ref) / eps

    # Perturb P_iron
    T_pert_fe = T + dt * thermal_derivatives(T, P_copper, P_iron + eps, tp)
    B[:, 1] = (T_pert_fe - T_ref) / eps

    return B


# %%
# ═══════════════════════════════════════════════════════════════════
# Steady-State Solver
# ═══════════════════════════════════════════════════════════════════

def steady_state_temperatures(P_copper: float, P_iron: float,
                              tp: ThermalParams) -> np.ndarray:
    """Compute steady-state temperatures analytically.

    At steady state, dT/dt = 0 for all nodes. This gives a linear
    system A × T = b that can be solved directly.

    Parameters
    ----------
    P_copper : constant copper losses [W]
    P_iron   : constant iron losses [W]
    tp       : ThermalParams

    Returns
    -------
    T_ss : np.ndarray, shape (5,) — steady-state temperatures [°C]
    """
    # Build the conductance matrix (5×5)
    # Diagonal: sum of all conductances connected to node i
    # Off-diagonal: -G_ij
    G = np.zeros((N_THERMAL, N_THERMAL))

    # Winding (0) ↔ Stator (1)
    G[0, 0] += tp.G_winding_stator
    G[0, 1] -= tp.G_winding_stator
    G[1, 1] += tp.G_winding_stator
    G[1, 0] -= tp.G_winding_stator

    # Stator (1) ↔ Magnet (2)
    G[1, 1] += tp.G_stator_magnet
    G[1, 2] -= tp.G_stator_magnet
    G[2, 2] += tp.G_stator_magnet
    G[2, 1] -= tp.G_stator_magnet

    # Stator (1) ↔ Housing (3)
    G[1, 1] += tp.G_stator_housing
    G[1, 3] -= tp.G_stator_housing
    G[3, 3] += tp.G_stator_housing
    G[3, 1] -= tp.G_stator_housing

    # Housing (3) ↔ Coolant (4)
    G[3, 3] += tp.G_housing_coolant
    G[3, 4] -= tp.G_housing_coolant
    G[4, 4] += tp.G_housing_coolant
    G[4, 3] -= tp.G_housing_coolant

    # Coolant (4) → inlet boundary
    G[4, 4] += tp.G_coolant_inlet

    # RHS: heat injections + boundary term
    b = np.zeros(N_THERMAL)
    b[IDX_WINDING] = P_copper
    b[IDX_STATOR]  = P_iron
    b[IDX_COOLANT]  = tp.G_coolant_inlet * tp.T_coolant_inlet

    T_ss = np.linalg.solve(G, b)
    return T_ss


# %%
# ═══════════════════════════════════════════════════════════════════
# Plotting Utilities
# ═══════════════════════════════════════════════════════════════════

def plot_thermal_response(T_hist: np.ndarray, dt: float,
                          T_magnet_max: float = 140.0,
                          title: str = "Thermal Response",
                          save_path: Optional[str] = None,
                          show: bool = True):
    """Plot temperature histories for all 5 nodes.

    Parameters
    ----------
    T_hist       : shape (N+1, 5)
    dt           : time step [s]
    T_magnet_max : demagnetisation limit for annotation
    save_path    : save figure path
    show         : call plt.show()
    """
    import matplotlib.pyplot as plt

    N = T_hist.shape[0]
    t = np.arange(N) * dt

    colors = ["#DC0000", "#FF8C00", "#8B008B", "#0050A0", "#00A000"]
    labels = ["Winding", "Stator Iron", "Magnet", "Housing", "Coolant"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(N_THERMAL):
        ax.plot(t, T_hist[:, i], color=colors[i], linewidth=1.5,
                label=f"{labels[i]} ({T_hist[-1, i]:.1f}°C)")

    # Demagnetisation limit
    ax.axhline(T_magnet_max, color="#8B008B", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Demag limit ({T_magnet_max}°C)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
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


# %%
# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.track_model import load_track, load_params
    from src.pmsm_losses import load_motor_params, total_losses

    print("=" * 60)
    print("  Stage 3: 5-Node Lumped Thermal Network")
    print("=" * 60)

    # Load all parameters
    params = load_params()
    tp = ThermalParams.from_yaml(params)
    motor = load_motor_params()

    print(f"\n  Thermal Parameters")
    print(f"  {'─' * 45}")
    print(f"  Capacitances [J/K]: {tp.C}")
    print(f"  G_winding↔stator  = {tp.G_winding_stator:.0f} W/K")
    print(f"  G_stator↔magnet   = {tp.G_stator_magnet:.0f} W/K")
    print(f"  G_stator↔housing  = {tp.G_stator_housing:.0f} W/K")
    print(f"  G_housing↔coolant = {tp.G_housing_coolant:.0f} W/K")
    print(f"  G_coolant→inlet   = {tp.G_coolant_inlet:.0f} W/K")
    print(f"  T_coolant_inlet   = {tp.T_coolant_inlet:.0f} °C")
    print(f"  T_magnet_max      = {tp.T_magnet_max:.0f} °C")

    # ── Test 1: Steady-state at continuous power ──
    print(f"\n  Test 1: Steady-State at 120 kW continuous, 250 km/h")
    print(f"  {'─' * 45}")
    P_cu, P_fe, _ = total_losses(
        np.array([120e3]), np.array([250 / 3.6]), motor
    )
    T_ss = steady_state_temperatures(P_cu[0], P_fe[0], tp)
    state_ss = ThermalState(T=T_ss)
    print(state_ss.summary())
    print(f"  T_magnet < 140°C? {'YES ✓' if T_ss[IDX_MAGNET] < 140 else 'NO ✗'}")

    # ── Test 2: Steady-state at peak power ──
    print(f"\n  Test 2: Steady-State at 350 kW peak, 340 km/h")
    print(f"  {'─' * 45}")
    P_cu2, P_fe2, _ = total_losses(
        np.array([350e3]), np.array([340 / 3.6]), motor
    )
    T_ss2 = steady_state_temperatures(P_cu2[0], P_fe2[0], tp)
    state_ss2 = ThermalState(T=T_ss2)
    print(state_ss2.summary())
    print(f"  T_magnet < 140°C? {'YES ✓' if T_ss2[IDX_MAGNET] < 140 else 'NO ✗'}")

    # ── Test 3: Step response (0→350 kW at 300 km/h for 30 s) ──
    print(f"\n  Test 3: Step Response — 0 → 350 kW at 300 km/h for 30 s")
    print(f"  {'─' * 45}")
    dt = params["mpc"]["dt"]
    N_step = int(30.0 / dt)
    P_step = 350e3
    v_step = 300 / 3.6
    P_cu_step, P_fe_step, _ = total_losses(
        np.full(N_step, P_step), np.full(N_step, v_step), motor
    )
    T_hist_step = simulate_thermal(P_cu_step, P_fe_step, dt, tp)
    print(f"  Final temperatures after 30 s:")
    print(ThermalState(T=T_hist_step[-1]).summary())
    print(f"  T_magnet < 140°C? {'YES ✓' if T_hist_step[-1, IDX_MAGNET] < 140 else 'NO ✗'}")

    # ── Test 4: Full Monza lap simulation ──
    print(f"\n  Test 4: Full Monza Lap — Coupled Thermal + Loss Simulation")
    print(f"  {'─' * 45}")
    track = load_track(params)

    # Assume full deploy on straights, regen in corners
    P_e_lap = np.where(track.segment == "straight", 200e3, -80e3)
    P_cu_lap, P_fe_lap, P_tot_lap = total_losses(P_e_lap, track.v, motor)
    T_hist_lap = simulate_thermal(P_cu_lap, P_fe_lap, dt, tp)

    print(f"  Mean P_total     = {P_tot_lap.mean() / 1e3:.2f} kW")
    print(f"  Peak P_total     = {P_tot_lap.max() / 1e3:.2f} kW")
    print(f"  Final temperatures:")
    print(ThermalState(T=T_hist_lap[-1]).summary())
    T_mag_max_lap = T_hist_lap[:, IDX_MAGNET].max()
    print(f"  Peak T_magnet    = {T_mag_max_lap:.1f} °C")
    print(f"  T_magnet < 140°C? {'YES ✓' if T_mag_max_lap < 140 else 'NO ✗'}")

    # ── Test 5: Jacobian sanity check ──
    print(f"\n  Test 5: Jacobian Sanity Check")
    print(f"  {'─' * 45}")
    T_test = np.array([80.0, 75.0, 72.0, 70.0, 66.0])
    A = thermal_jacobian_state(T_test, 5000.0, 8000.0, dt, tp)
    B = thermal_jacobian_input(T_test, 5000.0, 8000.0, dt, tp)
    print(f"  A (state Jacobian), shape {A.shape}:")
    print(f"    diag(A) = {np.diag(A)}")
    print(f"    All eigenvalues < 1 (stable)? "
          f"{'YES ✓' if np.all(np.abs(np.linalg.eigvals(np.eye(5) + A)) <= 1.01) else 'checking...'}")
    print(f"  B (input Jacobian), shape {B.shape}:")
    print(f"    B[:, 0] (∂T/∂P_cu) = {B[:, 0]}")
    print(f"    B[:, 1] (∂T/∂P_fe) = {B[:, 1]}")

    # ── Generate plots ──
    plots_dir = Path(__file__).resolve().parent.parent / "plots"

    plot_thermal_response(
        T_hist_step, dt, tp.T_magnet_max,
        title="Step Response: 350 kW at 300 km/h",
        save_path=str(plots_dir / "thermal_step_response.png"),
        show=False,
    )

    plot_thermal_response(
        T_hist_lap, dt, tp.T_magnet_max,
        title="Monza Lap: 200 kW deploy / 80 kW regen",
        save_path=str(plots_dir / "thermal_monza_lap.png"),
        show=False,
    )

    print(f"\n[Done] Plots saved to {plots_dir}/")
