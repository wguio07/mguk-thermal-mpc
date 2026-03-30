# %%
"""
pmsm_losses.py — PMSM Copper + Iron Loss Model
════════════════════════════════════════════════
Stage 2 of MGU-K Thermal-Constrained ERS Deployment Optimiser

Models the electromagnetic losses in a high-performance permanent
magnet synchronous motor (PMSM) at the 350 kW class for the 2026
F1 regulations.

Physics chain:
    P_e (electrical power request)
    → I_s (stator current, from torque/power inversion)
    → P_copper = (3/2) × R_s × I_s²
    → P_iron   = k_h·f·B_pk^α + k_e·(f·B_pk)²   (Steinmetz)
    → P_total  = P_copper + P_iron
    → feeds into 5-node thermal network (Stage 3)

The iron loss depends on electrical frequency (hence motor speed),
while the copper loss depends on current (hence torque/power). This
speed–torque coupling is why deployment and thermal management are
fundamentally the same optimisation problem.

Author: Wolfgang Guio
Project: Ferrari F1 Engineering Academy 2026
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

# %%
# ═══════════════════════════════════════════════════════════════════
# Motor Parameters Container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MotorParams:
    """Holds all PMSM electrical and loss parameters.

    Loaded from motor_params.yaml and provides derived quantities
    (e.g. max torque, base speed) used throughout the loss model.
    """
    # Electrical
    R_s: float          # [Ω] stator resistance per phase
    psi_m: float        # [Wb] permanent magnet flux linkage
    pole_pairs: int     # [-] number of pole pairs
    P_peak: float       # [W] peak electrical power
    P_continuous: float # [W] continuous electrical power

    # Steinmetz iron-loss coefficients
    k_h: float          # hysteresis coefficient
    k_e: float          # eddy-current coefficient
    alpha: float        # Steinmetz exponent

    # Magnetic saturation coefficients
    # At high stator current, iron saturation reduces effective ψ_m,
    # forcing more current for the same torque → higher copper losses.
    k_sat: float = 0.08   # flux reduction factor [-]
    I_sat: float = 400.0  # saturation reference current [A]

    @property
    def tau_max(self) -> float:
        """Maximum motor torque [Nm], estimated from peak power at base speed.

        For a 350 kW motor with typical base speed ~10,000 RPM:
            τ_max ≈ P_peak / ω_base
        We estimate base speed as the point where back-EMF equals
        a typical DC bus voltage (~800 V for F1 hybrid).
        """
        # V_dc ≈ 800 V → peak phase voltage ≈ V_dc / sqrt(3)
        V_phase_peak = 800.0 / np.sqrt(3)
        # Back-EMF: E = ω_e × ψ_m = (p × ω_m) × ψ_m
        # At base speed: E = V_phase_peak
        # → ω_m_base = V_phase_peak / (p × ψ_m)
        omega_m_base = V_phase_peak / (self.pole_pairs * self.psi_m)
        return self.P_peak / omega_m_base

    @property
    def omega_m_base(self) -> float:
        """Base mechanical speed [rad/s] — above this, field weakening."""
        V_phase_peak = 800.0 / np.sqrt(3)
        return V_phase_peak / (self.pole_pairs * self.psi_m)

    @classmethod
    def from_yaml(cls, params: dict) -> "MotorParams":
        """Construct from parsed motor_params.yaml dict."""
        e = params["electrical"]
        s = params["steinmetz"]
        sat = params.get("saturation", {})
        return cls(
            R_s=e["R_s"],
            psi_m=e["psi_m"],
            pole_pairs=e["pole_pairs"],
            P_peak=e["P_peak"],
            P_continuous=e["P_continuous"],
            k_h=s["k_h"],
            k_e=s["k_e"],
            alpha=s["alpha"],
            k_sat=sat.get("k_sat", 0.08),
            I_sat=sat.get("I_sat", 400.0),
        )


def load_motor_params(params_path: Optional[str] = None) -> MotorParams:
    """Load motor parameters from YAML.

    Parameters
    ----------
    params_path : str, optional
        Path to motor_params.yaml. If None, auto-discovers from
        this file's location.

    Returns
    -------
    MotorParams
    """
    if params_path is None:
        params_path = Path(__file__).resolve().parent.parent / "params" / "motor_params.yaml"
    # Use UTF-8 (with optional BOM) so Unicode comments parse on Windows.
    with open(params_path, "r", encoding="utf-8-sig") as f:
        params = yaml.safe_load(f)
    return MotorParams.from_yaml(params)


# %%
# ═══════════════════════════════════════════════════════════════════
# Magnetic Saturation Model
# ═══════════════════════════════════════════════════════════════════

def saturation_factor(I_s: np.ndarray, motor: MotorParams) -> np.ndarray:
    """Effective flux-linkage reduction factor due to iron saturation.

    At high stator current, the stator iron approaches magnetic saturation.
    The peak flux that the iron can channel decreases, reducing the effective
    permanent magnet flux linkage ψ_m_eff = ψ_m × sat(I_s).

    Because τ = (3/2) × p × ψ_m_eff × I_s, a lower ψ_m_eff means the motor
    must draw MORE current to achieve the same torque:
        I_s_eff = I_s_nominal / sat(I_s_nominal)

    This raises copper losses (I²R) and hence magnet temperature — the key
    physical consequence of saturation for thermal management.

    Model (first-order quadratic approximation):
        sat(I_s) = 1 − k_sat × (I_s / I_sat)²
        Clamped to [0.5, 1.0].

    Parameters
    ----------
    I_s   : np.ndarray  — nominal (unsaturated) stator RMS current [A]
    motor : MotorParams — contains k_sat and I_sat

    Returns
    -------
    sat : np.ndarray   — factor in [0.5, 1.0]. 1.0 = no saturation.

    Notes
    -----
    Calibrated: k_sat=0.08, I_sat=400 A → ~8% flux reduction at peak
    current, causing ~17% more copper losses at full deployment.
    Literature: Morimoto et al. (2006), Bianchi & Bolognani (2002).
    """
    sat = 1.0 - motor.k_sat * (I_s / motor.I_sat) ** 2
    return np.maximum(sat, 0.5)   # physical lower bound (50% residual flux)


# %%
# ═══════════════════════════════════════════════════════════════════
# Current Inversion: P_e → I_s
# ═══════════════════════════════════════════════════════════════════

def power_to_current(P_e: np.ndarray, v: np.ndarray,
                     motor: MotorParams) -> np.ndarray:
    """Invert electrical power to stator RMS current.

    For a PMSM under id=0 control (maximum torque per ampere):
        P_mech = τ × ω_m  and  τ = (3/2) × p × ψ_m × I_s
        → I_s = P_mech / ((3/2) × p × ψ_m × ω_m)

    The mechanical power includes copper losses, so we iterate:
        P_mech = P_e - P_copper(I_s)

    For computational efficiency in the MPC, we use a direct
    first-order approximation (neglecting the loss feedback for
    the current estimate, since P_copper << P_e at these power
    levels). The full loss is then computed from the resulting I_s.

    We derive ω_m from vehicle speed using a fixed overall gear
    ratio. For Monza, a representative ratio maps ~340 km/h at
    ~12,000 RPM → k_gear ≈ 2π×12000/60 / (340/3.6) ≈ 13.3 rad/s
    per m/s.

    Parameters
    ----------
    P_e : np.ndarray
        Electrical power request [W]. Positive = deploy, negative = regen.
    v : np.ndarray
        Vehicle speed [m/s].
    motor : MotorParams

    Returns
    -------
    I_s : np.ndarray
        Stator RMS current [A]. Always non-negative.
    """
    p = motor.pole_pairs
    psi_m = motor.psi_m

    # Vehicle speed → motor mechanical speed
    # k_gear maps vehicle speed to motor speed
    # At 340 km/h (~94.4 m/s) → ~12,000 RPM (~1257 rad/s)
    k_gear = 1257.0 / 94.4  # ≈ 13.3 rad·s⁻¹ per m·s⁻¹
    omega_m = k_gear * np.maximum(v, 5.0)  # floor at 5 m/s to avoid div/0

    # Torque from power: τ = P / ω_m
    tau = P_e / omega_m  # [Nm], can be negative (regen)

    # PMSM torque equation (id=0 control):
    #   τ = (3/2) × p × ψ_m × I_q
    # Step 1: nominal current assuming no saturation
    I_s_0 = np.abs(tau) / (1.5 * p * psi_m)

    # Step 2: magnetic saturation reduces effective ψ_m.
    # Effective flux: ψ_m_eff = ψ_m × sat(I_s_0)
    # To produce the same torque with less flux, more current is needed:
    #   I_s_eff = I_s_0 / sat(I_s_0)
    # This is a first-order approximation (avoids the fixed-point iteration
    # that the exact solution would require).
    sat = saturation_factor(I_s_0, motor)
    I_s_eff = I_s_0 / sat   # always ≥ I_s_0 (saturation only increases losses)

    return I_s_eff


def speed_to_electrical_freq(v: np.ndarray,
                             motor: MotorParams) -> np.ndarray:
    """Convert vehicle speed to electrical frequency [Hz].

    f_e = (p × ω_m) / (2π)

    Parameters
    ----------
    v : np.ndarray      — vehicle speed [m/s]
    motor : MotorParams

    Returns
    -------
    f_e : np.ndarray    — electrical frequency [Hz]
    """
    k_gear = 1257.0 / 94.4
    omega_m = k_gear * np.maximum(v, 5.0)
    omega_e = motor.pole_pairs * omega_m
    f_e = omega_e / (2.0 * np.pi)
    return f_e


# %%
# ═══════════════════════════════════════════════════════════════════
# Copper Losses
# ═══════════════════════════════════════════════════════════════════

def copper_losses(I_s: np.ndarray, motor: MotorParams) -> np.ndarray:
    """Compute copper (ohmic) losses in the stator windings.

    P_copper = (3/2) × R_s × I_s²

    This is the dominant loss mechanism at high torque / low speed.
    The factor 3/2 comes from the Park transform (3 phases → dq frame,
    assuming balanced operation with id=0).

    Parameters
    ----------
    I_s : np.ndarray    — stator RMS current [A]
    motor : MotorParams

    Returns
    -------
    P_copper : np.ndarray  — copper losses [W]
    """
    return 1.5 * motor.R_s * I_s**2


# %%
# ═══════════════════════════════════════════════════════════════════
# Iron Losses (Steinmetz Equation)
# ═══════════════════════════════════════════════════════════════════

def iron_losses(f_e: np.ndarray, motor: MotorParams,
                B_pk: float = 1.5) -> np.ndarray:
    """Compute iron losses using the modified Steinmetz equation.

    P_iron = k_h × f × B_pk^α  +  k_e × (f × B_pk)²
             ─────────────────     ──────────────────
             hysteresis term        eddy-current term

    The peak flux density B_pk is assumed approximately constant
    across the operating range (typical for automotive PMSMs with
    optimised slot geometry). At very high field-weakening speeds
    B_pk would decrease, but for Monza's speed range this is a
    reasonable assumption.

    Iron losses are the dominant loss mechanism at high speed / low
    torque — the opposite regime from copper losses. This is what
    makes the speed–power coupling so important for deployment
    optimisation.

    Parameters
    ----------
    f_e : np.ndarray       — electrical frequency [Hz]
    motor : MotorParams
    B_pk : float           — peak flux density [T] (default 1.5 T)

    Returns
    -------
    P_iron : np.ndarray    — iron losses [W]
    """
    P_hyst = motor.k_h * f_e * B_pk**motor.alpha
    P_eddy = motor.k_e * (f_e * B_pk)**2
    return P_hyst + P_eddy


# %%
# ═══════════════════════════════════════════════════════════════════
# Combined Loss Model (Main Interface)
# ═══════════════════════════════════════════════════════════════════

def total_losses(P_e: np.ndarray, v: np.ndarray,
                 motor: MotorParams,
                 B_pk: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute total PMSM losses from electrical power and vehicle speed.

    This is the main interface used by the thermal network (Stage 3)
    and the MPC controller (Stage 4). Given a power command P_e and
    vehicle speed v, it returns the heat that must be dissipated.

    A mild power-dependent flux density B_pk_eff is used to capture the
    increase in air-gap flux under heavy loading (cross-saturation effect).
    This keeps the MPC thermal predictions accurate across the full
    deployment range (from regen to 350 kW peak).

    Parameters
    ----------
    P_e : np.ndarray       — electrical power request [W]
    v : np.ndarray         — vehicle speed [m/s]
    motor : MotorParams
    B_pk : float           — nominal peak flux density [T] (used as base)

    Returns
    -------
    P_copper : np.ndarray  — copper losses [W] (→ winding node)
    P_iron : np.ndarray    — iron losses [W] (→ stator iron node)
    P_total : np.ndarray   — total losses [W]
    """
    # Mild load-dependent B_pk: rises from B_pk_base at no-load to B_pk at peak
    # B_pk_eff = B_pk_base + (B_pk - B_pk_base) × |P_e| / P_peak
    # With B_pk_base = 1.0 T and B_pk = 1.5 T:
    #   At P_e = 0   → B_pk_eff = 1.0 T  (base flux from magnets only)
    #   At P_e = 350 kW → B_pk_eff = 1.5 T (cross-saturation at full load)
    B_pk_eff = 1.0 + (B_pk - 1.0) * (np.abs(P_e) / motor.P_peak)

    I_s = power_to_current(P_e, v, motor)
    f_e = speed_to_electrical_freq(v, motor)

    P_cu = copper_losses(I_s, motor)
    P_fe = iron_losses(f_e, motor, B_pk_eff)
    P_tot = P_cu + P_fe

    return P_cu, P_fe, P_tot


def loss_at_operating_point(P_e: float, v: float,
                            motor: MotorParams,
                            B_pk: float = 1.5) -> dict:
    """Compute losses at a single operating point (scalar interface).

    Useful for debugging and interactive exploration.

    Parameters
    ----------
    P_e : float   — electrical power [W]
    v : float     — vehicle speed [m/s]
    motor : MotorParams
    B_pk : float  — peak flux density [T]

    Returns
    -------
    dict with keys: I_s, f_e, omega_m, tau, P_copper, P_iron,
                    P_total, efficiency
    """
    P_e_arr = np.array([P_e])
    v_arr = np.array([v])

    I_s = power_to_current(P_e_arr, v_arr, motor)[0]
    f_e = speed_to_electrical_freq(v_arr, motor)[0]

    k_gear = 1257.0 / 94.4
    omega_m = k_gear * max(v, 5.0)
    tau = P_e / omega_m

    P_cu = copper_losses(np.array([I_s]), motor)[0]
    P_fe = iron_losses(np.array([f_e]), motor, B_pk)[0]
    P_tot = P_cu + P_fe

    # Efficiency: η = P_mech / (P_mech + P_loss)  for deploy
    #             η = P_mech / (P_mech - P_loss)  for regen (not computed here)
    if abs(P_e) > 1.0:
        eta = abs(P_e) / (abs(P_e) + P_tot)
    else:
        eta = 0.0

    return {
        "I_s": I_s,
        "f_e": f_e,
        "omega_m": omega_m,
        "tau": tau,
        "P_copper": P_cu,
        "P_iron": P_fe,
        "P_total": P_tot,
        "efficiency": eta,
    }


# %%
# ═══════════════════════════════════════════════════════════════════
# Plotting Utilities
# ═══════════════════════════════════════════════════════════════════

def _analytical_losses(P_e: np.ndarray, v: np.ndarray,
                        motor: "MotorParams",
                        B_pk: float = 1.5):
    """Analytical loss computation for motor-map visualisation.

    Uses:
      • Fixed B_pk (independent of power) → iron losses purely speed-dependent,
        appearing as flat horizontal lines on P_e-axis plots.
      • No magnetic-saturation correction on stator current → copper losses
        follow the clean I_s = P/(τ·ω) relationship, giving the physically
        intuitive U-shaped loss curves.

    These simplifications are appropriate for motor-map plots where the goal
    is to show the fundamental loss mechanisms, not the exact MPC operating
    point. Use total_losses() for the MPC/thermal simulation.
    """
    p     = motor.pole_pairs
    psi_m = motor.psi_m
    k_gear = 1257.0 / 94.4
    omega_m = k_gear * np.maximum(v, 5.0)

    # Current without saturation correction (clean theoretical map)
    tau = P_e / omega_m
    I_s = np.abs(tau) / (1.5 * p * psi_m)

    f_e  = speed_to_electrical_freq(v, motor)
    P_cu = copper_losses(I_s, motor)
    P_fe = iron_losses(f_e, motor, B_pk)    # fixed B_pk — purely speed-dependent
    return P_cu, P_fe, P_cu + P_fe


def plot_loss_map(motor: "MotorParams", save_path: Optional[str] = None,
                  show: bool = True):
    """Plot loss maps: P_copper, P_iron, P_total vs electrical power at various speeds.

    Iron losses are purely speed-dependent (fixed B_pk = 1.5 T), so they
    appear as flat horizontal lines vs power — the analytically correct
    picture for a PMSM below field-weakening.

    Parameters
    ----------
    motor     : MotorParams
    save_path : if given, save figure here
    show      : call plt.show()
    """
    import matplotlib.pyplot as plt

    speeds_kmh = [100, 150, 200, 250, 300, 340]
    P_range = np.linspace(0, motor.P_peak, 500)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PMSM Loss Maps vs Electrical Power", fontsize=13, fontweight="bold")

    labels = ["Copper Losses", "Iron Losses", "Total Losses"]
    for ax, label in zip(axes, labels):
        ax.set_title(label)
        ax.set_xlabel("Electrical Power [kW]")
        ax.set_ylabel("Loss [kW]")
        ax.grid(True, alpha=0.3)

    for v_kmh in speeds_kmh:
        v = v_kmh / 3.6
        v_arr = np.full_like(P_range, v)
        P_cu, P_fe, P_tot = _analytical_losses(P_range, v_arr, motor)

        axes[0].plot(P_range / 1e3, P_cu / 1e3, label=f"{v_kmh} km/h")
        axes[1].plot(P_range / 1e3, P_fe / 1e3, label=f"{v_kmh} km/h")
        axes[2].plot(P_range / 1e3, P_tot / 1e3, label=f"{v_kmh} km/h")

    for ax in axes:
        ax.legend(fontsize=7, loc="upper left")

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


def plot_loss_vs_speed(motor: "MotorParams", save_path: Optional[str] = None,
                       show: bool = True):
    """Plot losses vs vehicle speed at fixed power levels.

    Shows the fundamental trade-off: copper losses dominate at high torque /
    low speed; iron losses dominate at high speed / low torque.

    Iron losses are purely speed-dependent (fixed B_pk), so all power-level
    curves coincide in the iron-loss panel — clearly revealing the speed
    dependence as independent from the deployment level.

    Parameters
    ----------
    motor     : MotorParams
    save_path : if given, save figure here
    show      : call plt.show()
    """
    import matplotlib.pyplot as plt

    v_range = np.linspace(100 / 3.6, 360 / 3.6, 500)   # 100–360 km/h in m/s
    power_levels_kW = [50, 100, 200, 300, 350]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PMSM Losses vs Vehicle Speed", fontsize=13, fontweight="bold")

    labels = ["Copper Losses", "Iron Losses", "Total Losses"]
    for ax, label in zip(axes, labels):
        ax.set_title(label)
        ax.set_xlabel("Speed [km/h]")
        ax.set_ylabel("Loss [kW]")
        ax.grid(True, alpha=0.3)

    for P_kW in power_levels_kW:
        P_e  = P_kW * 1e3
        P_arr = np.full_like(v_range, P_e)
        P_cu, P_fe, P_tot = _analytical_losses(P_arr, v_range, motor)

        axes[0].plot(v_range * 3.6, P_cu / 1e3, label=f"{P_kW} kW")
        axes[1].plot(v_range * 3.6, P_fe / 1e3, label=f"{P_kW} kW")
        axes[2].plot(v_range * 3.6, P_tot / 1e3, label=f"{P_kW} kW")

    for ax in axes:
        ax.legend(fontsize=7, loc="upper left")

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


def plot_efficiency_map(motor: "MotorParams", save_path: Optional[str] = None,
                        show: bool = True):
    """Plot motor efficiency as a 2D contourf heatmap (speed × power).

    Efficiency is computed as η = |P_e| / (|P_e| + P_loss) using a
    fixed B_pk = 1.5 T (analytical motor map).

    Parameters
    ----------
    motor     : MotorParams
    save_path : if given, save figure here
    show      : call plt.show()
    """
    import matplotlib.pyplot as plt

    v_range = np.linspace(100 / 3.6, 360 / 3.6, 200)   # m/s  (100–360 km/h)
    P_range = np.linspace(10e3, 350e3, 200)              # W

    V, P = np.meshgrid(v_range, P_range)
    P_cu, P_fe, P_tot = _analytical_losses(P.ravel(), V.ravel(), motor)
    P_tot_2d = P_tot.reshape(V.shape)

    # Efficiency: η = |P_e| / (|P_e| + P_loss)
    eta = np.abs(P) / (np.abs(P) + P_tot_2d) * 100   # [%]

    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.arange(80, 100.5, 0.5)
    cf   = ax.contourf(V * 3.6, P / 1e3, eta, levels=levels, cmap="RdYlGn")
    ax.contour(V * 3.6, P / 1e3, eta, levels=[90, 92, 94, 96, 98],
               colors="black", linewidths=0.5)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Efficiency [%]")

    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Electrical Power [kW]")
    ax.set_title("MGU-K Efficiency Map", fontweight="bold")
    ax.grid(True, alpha=0.2)

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
    print("  Stage 2: PMSM Loss Model — Copper + Iron Losses")
    print("=" * 60)

    motor = load_motor_params()

    # ── Print derived quantities ──
    print(f"\n  Motor Parameters")
    print(f"  {'─' * 45}")
    print(f"  R_s          = {motor.R_s * 1e3:.1f} mΩ")
    print(f"  ψ_m          = {motor.psi_m:.3f} Wb")
    print(f"  Pole pairs   = {motor.pole_pairs}")
    print(f"  P_peak       = {motor.P_peak / 1e3:.0f} kW")
    print(f"  ω_m_base     = {motor.omega_m_base:.0f} rad/s "
          f"({motor.omega_m_base * 60 / (2 * np.pi):.0f} RPM)")
    print(f"  τ_max (est.) = {motor.tau_max:.1f} Nm")

    # ── Spot-check a few operating points ──
    print(f"\n  Operating Point Checks")
    print(f"  {'─' * 45}")

    test_points = [
        (350000, 94.0, "Peak deploy at v_max (350 kW, 340 km/h)"),
        (350000, 40.0, "Peak deploy at low speed (350 kW, 144 km/h)"),
        (120000, 70.0, "Continuous deploy (120 kW, 252 km/h)"),
        (-120000, 80.0, "Regen braking (-120 kW, 288 km/h)"),
        (50000, 50.0, "Light deploy (50 kW, 180 km/h)"),
    ]

    for P_e, v, desc in test_points:
        r = loss_at_operating_point(P_e, v, motor)
        print(f"\n  {desc}")
        print(f"    I_s     = {r['I_s']:.1f} A")
        print(f"    f_e     = {r['f_e']:.0f} Hz")
        print(f"    P_cu    = {r['P_copper'] / 1e3:.2f} kW")
        print(f"    P_fe    = {r['P_iron'] / 1e3:.2f} kW")
        print(f"    P_total = {r['P_total'] / 1e3:.2f} kW")
        print(f"    η       = {r['efficiency'] * 100:.1f}%")

    # ── Generate plots ──
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plot_loss_map(motor, save_path=str(plots_dir / "pmsm_loss_map.png"), show=False)
    plot_loss_vs_speed(motor, save_path=str(plots_dir / "pmsm_loss_vs_speed.png"), show=False)
    plot_efficiency_map(motor, save_path=str(plots_dir / "pmsm_efficiency_map.png"), show=False)

    print(f"\n[Done] Plots saved to {plots_dir}/")
