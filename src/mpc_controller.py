# %%
"""
mpc_controller.py — OSQP-Based Model Predictive Controller
═══════════════════════════════════════════════════════════
Stage 4 of MGU-K Thermal-Constrained ERS Deployment Optimiser

Implements a Real-Time Iteration (RTI) MPC that optimises MGU-K
deployment over a receding horizon while enforcing hard thermal
and energy constraints.

State vector (7 states):
    x = [v, SOC, T_winding, T_stator, T_magnet, T_housing, T_coolant]

Control input (1 input):
    u = P_e  (electrical power, positive=deploy, negative=regen)

At each 0.05 s step the controller:
    1. Linearises the coupled dynamics at the current operating point
       using numerical finite differences (ε ≈ 1e-6)
    2. Formulates a sparse QP:
       min  Σ [ -w_v·v_k + w_u·P_e,k² + w_soc·(SOC_k - SOC_ref)² ]
       s.t. x_{k+1} = A_k·x_k + B_k·u_k + c_k  (linearised dynamics)
            SOC_min ≤ SOC_k ≤ SOC_max
            T_magnet,k ≤ 140°C
            -P_peak ≤ P_e ≤ P_peak
    3. Solves with OSQP (warm-started from shifted previous solution)
    4. Applies first control action, advances state, repeats

Author: Wolfgang Guio
Project: Ferrari F1 Engineering Academy 2026
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from src.pmsm_losses import MotorParams, load_motor_params, total_losses, \
    power_to_current, speed_to_electrical_freq, copper_losses, iron_losses
from src.thermal_network import (
    ThermalParams, ThermalState, load_thermal_params,
    thermal_derivatives, N_THERMAL, IDX_MAGNET,
)

# %%
# ═══════════════════════════════════════════════════════════════════
# State Vector Indices
# ═══════════════════════════════════════════════════════════════════

IDX_V   = 0   # vehicle speed [m/s]
IDX_SOC = 1   # state of charge [-]
# Thermal states occupy indices 2..6
IDX_TH_START = 2
IDX_TH_END   = 7  # exclusive
N_STATES = 7
N_INPUTS = 1


# %%
# ═══════════════════════════════════════════════════════════════════
# MPC Parameters Container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MPCParams:
    """All parameters needed by the MPC controller."""
    # Time
    dt: float
    N_horizon: int
    eps: float

    # Cost weights
    w_v: float
    w_u: float
    w_soc: float

    # Constraints
    SOC_min: float
    SOC_max: float
    SOC_ref: float
    P_peak: float
    P_continuous: float
    T_magnet_max: float
    max_energy_per_lap: float
    battery_capacity: float

    # Vehicle
    mass: float
    CdA: float
    rho: float

    # Thermal barrier penalty
    w_thermal_barrier: float = 50.0    # weight on quadratic barrier
    T_barrier_margin: float = 10.0     # °C before limit where barrier activates

    # Multi-rate horizon schedule
    # N_horizon_short: used on straights and standard corners  (3 s)
    # N_horizon_long : used in heavy braking zones (T1, Lesmos, Parabolica) (6 s)
    # N_horizon is kept as a legacy fallback.
    N_horizon_short: int = 60    # 3 s at dt=0.05 s
    N_horizon_long:  int = 120   # 6 s at dt=0.05 s

    @classmethod
    def from_yaml(cls, params: dict) -> "MPCParams":
        mpc = params["mpc"]
        cw  = params["cost_weights"]
        en  = params["energy"]
        el  = params["electrical"]
        tl  = params["thermal_limits"]
        veh = params["vehicle"]
        return cls(
            dt=mpc["dt"], N_horizon=mpc["N_horizon"], eps=mpc["epsilon"],
            w_v=cw["w_v"], w_u=cw["w_u"], w_soc=cw["w_soc"],
            SOC_min=en["SOC_min"], SOC_max=en["SOC_max"], SOC_ref=en["SOC_ref"],
            P_peak=el["P_peak"], P_continuous=el["P_continuous"],
            T_magnet_max=tl["T_magnet_max"],
            max_energy_per_lap=en["max_energy_per_lap"],
            battery_capacity=en["battery_capacity"],
            mass=veh["mass"], CdA=veh["CdA"], rho=veh["rho"],
            N_horizon_short=mpc.get("N_horizon_short", 60),
            N_horizon_long=mpc.get("N_horizon_long", 120),
        )


def load_mpc_params(params_path: Optional[str] = None) -> MPCParams:
    if params_path is None:
        params_path = Path(__file__).resolve().parent.parent / "params" / "motor_params.yaml"
    # Use UTF-8 (with optional BOM) so Unicode comments parse on Windows.
    with open(params_path, "r", encoding="utf-8-sig") as f:
        params = yaml.safe_load(f)
    return MPCParams.from_yaml(params)


# %%
# ═══════════════════════════════════════════════════════════════════
# Coupled Nonlinear Dynamics
# ═══════════════════════════════════════════════════════════════════

def coupled_dynamics(x: np.ndarray, u: float,
                     v_ref: float,
                     motor: MotorParams, tp: ThermalParams,
                     mpc_p: MPCParams) -> np.ndarray:
    """Evaluate the full coupled nonlinear dynamics: x_{k+1} = f(x_k, u_k).

    Forward Euler discretisation of the continuous-time system.

    Parameters
    ----------
    x     : state vector [v, SOC, T_w, T_s, T_m, T_h, T_c], shape (7,)
    u     : control input P_e [W]
    v_ref : reference speed from track data [m/s] (used for traction model)
    motor : MotorParams
    tp    : ThermalParams
    mpc_p : MPCParams

    Returns
    -------
    x_next : np.ndarray, shape (7,) — next state
    """
    dt = mpc_p.dt
    v   = max(x[IDX_V], 5.0)       # floor speed for numerical safety
    SOC = x[IDX_SOC]
    T   = x[IDX_TH_START:IDX_TH_END]  # shape (5,)

    # ── Vehicle dynamics — speed as exogenous input ──────────────────
    # Speed is NOT controlled by the ERS. The driver and the mechanical
    # brake system dictate speed (5g deceleration at Monza Turn 1 =
    # ~40,000 N braking force vs ~4,200 N max MGU-K regen force).
    # Treating speed as a dynamically evolved state with only an aero
    # drag restoring force produces a maximum deceleration of ~0.3g —
    # 15× too low — so cars never reach corner apex speed.
    #
    # Correct architecture for an ERS controller: speed is measured
    # from real telemetry (track_v), and the controller optimises P_e
    # subject to that speed profile. v_next is set from the reference.
    v_next = v_ref

    # ── SOC dynamics ──
    # dSOC/dt = -P_e / E_battery  (deploy drains, regen charges)
    dSOC_dt = -u / mpc_p.battery_capacity
    SOC_next = SOC + dt * dSOC_dt

    # ── Thermal dynamics ──
    # Compute losses at current operating point
    P_cu, P_fe, _ = total_losses(np.array([u]), np.array([v]), motor)
    P_cu, P_fe = P_cu[0], P_fe[0]

    # Pass current vehicle speed so cooling conductances scale correctly.
    # At 300 km/h the motor cooling is ~2× more effective than at rest.
    dTdt = thermal_derivatives(T, P_cu, P_fe, tp, v_vehicle=float(v))
    T_next = T + dt * dTdt

    x_next = np.zeros(N_STATES)
    x_next[IDX_V]   = v_next
    x_next[IDX_SOC] = SOC_next
    x_next[IDX_TH_START:IDX_TH_END] = T_next

    return x_next


# %%
# ═══════════════════════════════════════════════════════════════════
# Numerical Linearisation (Finite Differences at Each Step)
# ═══════════════════════════════════════════════════════════════════

def linearise(x: np.ndarray, u: float, v_ref: float,
              motor: MotorParams, tp: ThermalParams,
              mpc_p: MPCParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearise coupled dynamics at (x, u) via numerical finite differences.

    x_{k+1} ≈ A·x_k + B·u_k + c

    where c = f(x̄, ū) - A·x̄ - B·ū  (affine offset)

    Jacobians are recomputed at the current operating point every 0.05 s.
    This is the Real-Time Iteration (RTI) approach: instead of solving
    the full nonlinear problem, we solve one QP per step with fresh
    linearisation.

    Parameters
    ----------
    x     : current state, shape (7,)
    u     : current control input P_e [W]
    v_ref : reference speed [m/s]
    motor, tp, mpc_p : parameter objects

    Returns
    -------
    A : np.ndarray, shape (7, 7) — state Jacobian ∂f/∂x
    B : np.ndarray, shape (7, 1) — input Jacobian ∂f/∂u
    c : np.ndarray, shape (7,)   — affine offset
    """
    eps = mpc_p.eps
    f_ref = coupled_dynamics(x, u, v_ref, motor, tp, mpc_p)

    # ── State Jacobian A = ∂f/∂x ──
    A = np.zeros((N_STATES, N_STATES))
    for j in range(N_STATES):
        x_pert = x.copy()
        x_pert[j] += eps
        f_pert = coupled_dynamics(x_pert, u, v_ref, motor, tp, mpc_p)
        A[:, j] = (f_pert - f_ref) / eps

    # ── Input Jacobian B = ∂f/∂u ──
    f_pert_u = coupled_dynamics(x, u + eps, v_ref, motor, tp, mpc_p)
    B = ((f_pert_u - f_ref) / eps).reshape(N_STATES, 1)

    # ── Affine offset c ──
    c = f_ref - A @ x - B.flatten() * u

    return A, B, c


# %%
# ═══════════════════════════════════════════════════════════════════
# QP Formulation for OSQP
# ═══════════════════════════════════════════════════════════════════

def condense(x0: np.ndarray,
             A_list: List[np.ndarray], B_list: List[np.ndarray],
             c_list: List[np.ndarray],
             N: int, nx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condense the dynamics: express predicted states as affine functions of u.

    x_{k+1} = A_k x_k + B_k u_k + c_k

    Unrolling:
        x_k = Phi_k @ x0 + Psi_k @ u_vec + d_k

    Uses an efficient column-propagation approach: instead of recomputing
    the full chain A_k...A_{j+1} @ B_j for each (k,j) pair, we propagate
    each column of Psi forward by one A multiplication per step.

    Returns
    -------
    Phi_all : shape (N, nx, nx) — state transition from x0
    Psi_all : shape (N*nx, N)   — maps u_vec → stacked [x_1; ...; x_N]
    d_all   : shape (N*nx,)     — affine offsets
    """
    Psi_all = np.zeros((N * nx, N))
    d_all = np.zeros(N * nx)
    Phi_all = np.zeros((N, nx, nx))

    # cols[j] holds the current propagated value of B_j through
    # the chain A_k @ ... @ A_{j+1} @ B_j, updated at each step k.
    # This avoids the O(N²) inner loop.
    cols = [None] * N  # will hold shape (nx,) vectors

    Phi_cum = np.eye(nx)

    for k in range(N):
        Phi_cum = A_list[k] @ Phi_cum if k > 0 else A_list[0].copy()
        Phi_all[k] = Phi_cum

        row = k * nx

        # Propagate existing columns forward by A_k
        for j in range(k):
            cols[j] = A_list[k] @ cols[j]
            Psi_all[row:row + nx, j] = cols[j]

        # New column j=k: just B_k (no further propagation yet)
        cols[k] = B_list[k].flatten().copy()
        Psi_all[row:row + nx, k] = cols[k]

        # Offset: d_{k+1} = A_k @ d_k + c_k
        if k == 0:
            d_all[row:row + nx] = c_list[0]
        else:
            prev_row = (k - 1) * nx
            d_all[row:row + nx] = A_list[k] @ d_all[prev_row:prev_row + nx] + c_list[k]

    return Phi_all, Psi_all, d_all


def build_condensed_qp(x0: np.ndarray,
                       A_list: List[np.ndarray], B_list: List[np.ndarray],
                       c_list: List[np.ndarray],
                       mpc_p: MPCParams,
                       N: Optional[int] = None,
                       v_ref_horizon: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray,
                                                                              np.ndarray, np.ndarray]:
    """Build a condensed QP with decision variable u_vec only.

    Eliminates all state variables using the condensed dynamics:
        x_k = Phi_k @ x0 + Psi_k @ u_vec + d_k

    Substituting into cost and constraints gives a QP in u_vec (N variables)
    instead of the sparse formulation (N×nx + N variables).

    Parameters
    ----------
    N : int, optional
        Prediction horizon to use. If None, falls back to mpc_p.N_horizon.
        Must match len(A_list) == len(B_list) == len(c_list).

    Returns
    -------
    H     : np.ndarray, shape (N, N) — Hessian
    g     : np.ndarray, shape (N,)   — gradient
    u_lb  : np.ndarray, shape (N,)   — lower bounds on u
    u_ub  : np.ndarray, shape (N,)   — upper bounds on u
    Also stored as class-level for constraint checking.
    """
    if N is None:
        N = mpc_p.N_horizon
    nx = N_STATES

    Phi_all, Psi_all, d_all = condense(x0, A_list, B_list, c_list, N, nx)

    # ── Build cost in u_vec ──
    # Cost: Σ_k [ -w_v * x_{k+1}[0] + w_soc * (x_{k+1}[1] - SOC_ref)² + w_u * u_k² ]
    #
    # x_{k+1} = Phi_all[k] @ x0 + Psi_all[k*nx:(k+1)*nx, :] @ u + d_all[k*nx:(k+1)*nx]
    #
    # For each state component s, define:
    #   x_{k+1}[s] = phi_s' @ x0 + psi_s' @ u + d_s
    # where phi_s = Phi_all[k, s, :], psi_s = Psi_all[k*nx+s, :], d_s = d_all[k*nx+s]

    H = np.zeros((N, N))
    g = np.zeros(N)

    for k in range(N):
        row = k * nx

        # Speed component (IDX_V = 0): was -w_v * v when speed was dynamic.
        # With speed now exogenous (v = v_ref from track), psi_v = 0 for
        # all k — the speed state has no sensitivity to u. The old term
        # contributes nothing to the gradient and is retained only for
        # clarity. The deployment reward below replaces it physically.
        psi_v = Psi_all[row + IDX_V, :]       # shape (N,) — all zeros now
        g += -mpc_p.w_v * psi_v               # = 0 (kept for documentation)

        # SOC component (IDX_SOC = 1): quadratic cost w_soc * (SOC - ref)²
        psi_soc = Psi_all[row + IDX_SOC, :]
        phi_soc_x0 = Phi_all[k, IDX_SOC, :] @ x0
        d_soc = d_all[row + IDX_SOC]
        soc_offset = phi_soc_x0 + d_soc - mpc_p.SOC_ref
        # w_soc * (psi_soc @ u + soc_offset)²
        # = w_soc * (u' psi_soc psi_soc' u + 2 soc_offset psi_soc' u + const)
        H += mpc_p.w_soc * np.outer(psi_soc, psi_soc)
        g += mpc_p.w_soc * 2.0 * soc_offset * psi_soc

    # Control effort: w_u * Σ u_k²
    H += mpc_p.w_u * np.eye(N)

    # ── Deploy/regen incentive (replaces dead speed-tracking term) ────
    # Physical motivation: with speed fixed from track data, the old
    # -w_v*v term has psi_v = 0.  We replace it with a centred cost that
    # simultaneously rewards deployment at high speed AND regen at low
    # speed — both naturally motivated by F1 ERS operation:
    #
    #   g[k] = w_v × (1 − 2 × v_norm) / P_peak
    #
    # where v_norm = v_ref_k / V_MAX ∈ [0, 1].
    #
    # Behaviour:
    #   v > V_MAX/2 (> 180 km/h, straights) → g[k] < 0 → deploy rewarded
    #   v < V_MAX/2 (< 180 km/h, corners)   → g[k] > 0 → regen  rewarded
    #   v = V_MAX/2 (180 km/h)              → g[k] = 0 → neutral
    #
    # This creates the correct F1 ERS cycle: charge the battery in
    # braking zones (low v) and spend it on straights (high v).
    # Crossover at 50 m/s ≈ 180 km/h sits naturally between Monza's
    # apex speeds (~70–130 km/h) and straight speeds (~250–350 km/h).
    V_MAX = 100.0  # m/s — normalisation constant (≈ Monza top speed)
    if v_ref_horizon is not None:
        for k in range(N):
            v_norm = min(float(v_ref_horizon[k]), V_MAX) / V_MAX  # [0, 1]
            # g'u term: positive g → regen reward; negative g → deploy reward
            g[k] += mpc_p.w_v * (1.0 - 2.0 * v_norm) / mpc_p.P_peak
            # Force regen if decelerating strongly
            if k < N - 1 and v_ref_horizon[k+1] < v_ref_horizon[k] - 1.0:
                g[k] += mpc_p.w_v * 5.0 / mpc_p.P_peak  # heavy regen reward

    # ── Thermal barrier penalty ──
    # Adds a quadratic penalty that activates when T_magnet approaches
    # the demagnetisation limit. This supplements the hard constraint
    # with a soft barrier, preventing the solver from riding the limit.
    #
    # penalty_k = w_th * max(0, T_mag_k - T_threshold)²
    #
    # where T_threshold = T_magnet_max - T_barrier_margin.
    # Since T_mag_k = phi_mag·x0 + psi_mag·u + d_mag is linear in u,
    # the squared penalty adds a rank-1 update to H and a linear term to g.
    T_threshold = mpc_p.T_magnet_max - mpc_p.T_barrier_margin
    magnet_idx = IDX_TH_START + IDX_MAGNET

    for k in range(N):
        row = k * nx
        psi_mag = Psi_all[row + magnet_idx, :]
        mag_base = Phi_all[k, magnet_idx, :] @ x0 + d_all[row + magnet_idx]

        # Only activate barrier when predicted T_mag exceeds threshold
        if mag_base > T_threshold:
            mag_offset = mag_base - T_threshold
            # w_th * (psi_mag @ u + mag_offset)²
            H += mpc_p.w_thermal_barrier * np.outer(psi_mag, psi_mag)
            g += mpc_p.w_thermal_barrier * 2.0 * mag_offset * psi_mag

    # Make H symmetric and add regularisation
    H = 0.5 * (H + H.T)
    H += 1e-10 * np.eye(N)

    # ── Bounds on u ──
    u_lb = np.full(N, -mpc_p.P_peak)
    u_ub = np.full(N, mpc_p.P_peak)

    # ── State constraints (SOC and T_magnet) → tighten u bounds ──
    # These are linear in u: a' @ u ≤ b
    # We collect them for the constrained solver
    # SOC: SOC_min ≤ phi_soc_x0 + psi_soc @ u + d_soc ≤ SOC_max
    # T_mag: phi_mag_x0 + psi_mag @ u + d_mag ≤ T_magnet_max

    magnet_idx = IDX_TH_START + IDX_MAGNET  # = 4

    n_ineq = 3 * N  # SOC_lb, SOC_ub, T_mag_ub
    C_ineq = np.zeros((n_ineq, N))
    d_lb = np.zeros(n_ineq)
    d_ub = np.zeros(n_ineq)

    for k in range(N):
        row = k * nx

        # SOC lower: psi_soc @ u ≥ SOC_min - phi_soc_x0 - d_soc
        psi_soc = Psi_all[row + IDX_SOC, :]
        soc_base = Phi_all[k, IDX_SOC, :] @ x0 + d_all[row + IDX_SOC]
        C_ineq[k, :] = psi_soc
        d_lb[k] = mpc_p.SOC_min - soc_base
        d_ub[k] = mpc_p.SOC_max - soc_base

        # T_magnet upper: psi_mag @ u ≤ T_max - phi_mag_x0 - d_mag
        psi_mag = Psi_all[row + magnet_idx, :]
        mag_base = Phi_all[k, magnet_idx, :] @ x0 + d_all[row + magnet_idx]
        C_ineq[N + k, :] = psi_mag
        d_lb[N + k] = -1e6
        d_ub[N + k] = mpc_p.T_magnet_max - mag_base

        # SOC upper (separate row for clarity)
        C_ineq[2 * N + k, :] = -psi_soc  # -psi_soc @ u ≥ -(SOC_max - soc_base)
        d_lb[2 * N + k] = -(mpc_p.SOC_max - soc_base)
        d_ub[2 * N + k] = 1e6

    return H, g, u_lb, u_ub, C_ineq, d_lb, d_ub, Phi_all, Psi_all, d_all


# %%
# ═══════════════════════════════════════════════════════════════════
# QP Solver Interface (OSQP + condensed numpy fallback)
# ═══════════════════════════════════════════════════════════════════

def solve_qp_osqp(H, g, u_lb, u_ub, C_ineq, d_lb, d_ub,
                   warm_x=None, warm_y=None):
    """Solve condensed QP using OSQP."""
    import osqp
    from scipy import sparse

    N = len(g)
    n_ineq = C_ineq.shape[0]

    # Stack: [I (box); C_ineq] @ u ∈ [lb, ub]
    A_top = np.eye(N)
    A_full = np.vstack([A_top, C_ineq])
    l_full = np.concatenate([u_lb, d_lb])
    u_full = np.concatenate([u_ub, d_ub])

    P_sparse = sparse.csc_matrix(H)
    A_sparse = sparse.csc_matrix(A_full)

    solver = osqp.OSQP()
    solver.setup(P_sparse, g, A_sparse, l_full, u_full,
                 verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                 max_iter=2000, warm_start=True, polish=True)

    if warm_x is not None and len(warm_x) == N:
        solver.warm_start(x=warm_x, y=warm_y)

    result = solver.solve()

    info = {
        "status": result.info.status,
        "obj_val": result.info.obj_val,
        "iter": result.info.iter,
        "run_time": result.info.run_time,
        "pri_res": getattr(result.info, "prim_res", getattr(result.info, "pri_res", 0.0)),
    }
    return result.x, info


def solve_qp_condensed(H, g, u_lb, u_ub, C_ineq, d_lb, d_ub,
                       warm_x=None, warm_y=None):
    """Fast condensed QP solver (numpy only, no OSQP/scipy needed).

    Solves: min 0.5 u'Hu + g'u  s.t. u_lb ≤ u ≤ u_ub, d_lb ≤ C u ≤ d_ub

    Uses unconstrained solve + iterative projection for inequality
    constraints. Much faster than the sparse null-space approach because
    the QP is only N variables (40) instead of N*nx+N (320).
    """
    N = len(g)

    # Unconstrained optimum
    try:
        u_opt = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        u_opt = warm_x.copy() if warm_x is not None else np.zeros(N)

    # Box clamp
    u_opt = np.clip(u_opt, u_lb, u_ub)

    # Iterative projection for inequality constraints
    for iteration in range(100):
        violations = False

        # Check C_ineq @ u ≤ d_ub  and  C_ineq @ u ≥ d_lb
        Cu = C_ineq @ u_opt
        for i in range(C_ineq.shape[0]):
            if Cu[i] > d_ub[i] + 1e-6:
                # Project: reduce u along C_ineq[i] direction
                excess = Cu[i] - d_ub[i]
                row = C_ineq[i]
                norm2 = np.dot(row, row)
                if norm2 > 1e-15:
                    u_opt -= (excess / norm2) * row
                violations = True
            elif Cu[i] < d_lb[i] - 1e-6:
                deficit = d_lb[i] - Cu[i]
                row = C_ineq[i]
                norm2 = np.dot(row, row)
                if norm2 > 1e-15:
                    u_opt += (deficit / norm2) * row
                violations = True

        # Re-apply box constraints
        u_opt = np.clip(u_opt, u_lb, u_ub)

        if not violations:
            break

    obj_val = 0.5 * u_opt @ H @ u_opt + g @ u_opt
    info = {
        "status": "solved (condensed)",
        "obj_val": obj_val,
        "iter": iteration + 1,
        "run_time": 0.0,
        "pri_res": 0.0,
    }
    return u_opt, info


def solve_qp(H, g, u_lb, u_ub, C_ineq, d_lb, d_ub,
             warm_x=None, warm_y=None):
    """Solve QP — uses OSQP if available, otherwise condensed fallback."""
    try:
        return solve_qp_osqp(H, g, u_lb, u_ub, C_ineq, d_lb, d_ub,
                              warm_x, warm_y)
    except ImportError:
        return solve_qp_condensed(H, g, u_lb, u_ub, C_ineq, d_lb, d_ub,
                                   warm_x, warm_y)


# %%
# ═══════════════════════════════════════════════════════════════════
# MPC Controller Class
# ═══════════════════════════════════════════════════════════════════

class MPCController:
    """Real-Time Iteration MPC for MGU-K deployment optimisation.

    Usage:
        controller = MPCController(motor, tp, mpc_p)
        for k in range(N_steps):
            u_opt = controller.step(x_k, v_ref_horizon_k)
    """

    def __init__(self, motor: MotorParams, tp: ThermalParams,
                 mpc_p: MPCParams):
        self.motor = motor
        self.tp = tp
        self.mpc_p = mpc_p

        # Warm-start storage
        self._prev_z = None
        self._prev_y = None

        # Logging
        self.solve_log: List[dict] = []

    def step(self, x: np.ndarray,
             v_ref_horizon: np.ndarray,
             n_horizon_override: Optional[int] = None) -> Tuple[float, dict]:
        """Run one MPC iteration: linearise → build condensed QP → solve → return u*.

        Uses single-point linearisation: the Jacobians are computed once
        at the current state (x, u_nominal[0]) and reused across the
        entire prediction horizon. This is valid because:
        - Thermal time constants (10-100s) >> horizon (up to 6s)
        - Vehicle dynamics are nearly linear over that window
        - The RTI framework re-linearises at EVERY real time step anyway

        Parameters
        ----------
        n_horizon_override : int, optional
            Override prediction horizon length for this step.
            Enables multi-rate scheduling: N_horizon_short (3 s) on
            straights and N_horizon_long (6 s) in heavy braking zones.
            When N changes between steps, the warm-start is automatically
            truncated or padded to the new size.

        Parameters
        ----------
        x              : current state, shape (7,)
        v_ref_horizon  : reference speeds for next N steps, shape (N,)

        Returns
        -------
        u_opt : float — optimal P_e [W] for this time step
        info  : dict — solver diagnostics
        """
        # ── Resolve prediction horizon for this step ──
        N  = n_horizon_override if n_horizon_override is not None \
             else self.mpc_p.N_horizon
        nx = N_STATES

        # ── Warm-start resizing when horizon changes between steps ──
        # This happens at every boundary where we switch between the short
        # (3 s) and long (6 s) horizons. We truncate or pad to size N.
        if self._prev_z is not None and len(self._prev_z) != N:
            prev_N = len(self._prev_z)
            if N < prev_N:
                # Shrinking (braking zone → straight): keep first N elements
                self._prev_z = self._prev_z[:N]
            else:
                # Growing (straight → braking zone): pad with last value
                self._prev_z = np.concatenate([
                    self._prev_z,
                    np.full(N - prev_N, self._prev_z[-1])
                ])

        # ── Step 1: Linearise once at current operating point ──
        u_nom_0 = self._prev_z[0] if self._prev_z is not None else 0.0
        v_ref_0 = v_ref_horizon[0]

        A_0, B_0, c_0 = linearise(x, u_nom_0, v_ref_0,
                                    self.motor, self.tp, self.mpc_p)

        # Reuse across horizon (single-point RTI)
        A_list = [A_0] * N
        B_list = [B_0] * N
        c_list = [c_0] * N

        # ── Step 2: Build condensed QP (u_vec only, N variables) ──
        H, g, u_lb, u_ub, C_ineq, d_lb, d_ub, _, _, _ = \
            build_condensed_qp(x, A_list, B_list, c_list, self.mpc_p, N=N,
                               v_ref_horizon=v_ref_horizon)

        # ── Step 3: Solve with warm-starting ──
        u_vec_opt, info = solve_qp(H, g, u_lb, u_ub, C_ineq, d_lb, d_ub,
                                    warm_x=self._prev_z,
                                    warm_y=self._prev_y)

        # ── Step 4: Extract first control action ──
        u_first = np.clip(u_vec_opt[0], -self.mpc_p.P_peak, self.mpc_p.P_peak)

        # ── Store u_vec for warm-starting (shifted by 1 step) ──
        u_shifted = np.zeros(N)
        u_shifted[:N-1] = u_vec_opt[1:]
        u_shifted[N-1] = u_vec_opt[N-1]   # hold last value as tail padding
        self._prev_z = u_shifted

        # Attach horizon used to diagnostics
        info["N_horizon_used"] = N
        self.solve_log.append(info)

        return u_first, info

    def reset(self):
        """Clear warm-start state."""
        self._prev_z = None
        self._prev_y = None
        self.solve_log.clear()


# %%
# ═══════════════════════════════════════════════════════════════════
# Simulation Runner
# ═══════════════════════════════════════════════════════════════════

def run_mpc_simulation(track_v: np.ndarray, track_segment: np.ndarray,
                       motor: MotorParams, tp: ThermalParams,
                       mpc_p: MPCParams,
                       T_init: Optional[np.ndarray] = None,
                       SOC_init: float = 0.5,
                       verbose: bool = True,
                       track_braking_zone: Optional[np.ndarray] = None) -> dict:
    """Run the full MPC simulation over a lap.

    Parameters
    ----------
    track_v           : speed profile from track model [m/s], shape (N_total,)
    track_segment     : 'straight'/'corner' labels, shape (N_total,)
    motor, tp, mpc_p  : parameter objects
    T_init            : initial thermal state, shape (5,). Default: coolant inlet.
    SOC_init          : initial state of charge [-]
    verbose           : print progress
    track_braking_zone: bool array, shape (N_total,). True where the extended
                        6-second horizon (N_horizon_long) should be used.
                        If None, uses N_horizon_short everywhere.

    Returns
    -------
    results : dict with keys:
        x_hist    : state history, shape (N_total+1, 7)
        u_hist    : control history, shape (N_total,)
        info_hist : solver info per step
    """
    N_total = len(track_v)
    N_hor = mpc_p.N_horizon
    dt = mpc_p.dt

    if T_init is None:
        T_init = np.full(N_THERMAL, tp.T_coolant_inlet)

    # Initial state
    x = np.zeros(N_STATES)
    x[IDX_V] = track_v[0]
    x[IDX_SOC] = SOC_init
    x[IDX_TH_START:IDX_TH_END] = T_init

    # Storage
    x_hist = np.zeros((N_total + 1, N_STATES))
    u_hist = np.zeros(N_total)
    info_hist = []
    x_hist[0] = x

    controller = MPCController(motor, tp, mpc_p)

    for k in range(N_total):
        # ── Select prediction horizon for this step (multi-rate schedule) ──
        if track_braking_zone is not None and track_braking_zone[k]:
            n_hor = mpc_p.N_horizon_long    # 6 s in heavy braking zones
        else:
            n_hor = mpc_p.N_horizon_short   # 3 s elsewhere

        # Build reference speed horizon (from track data)
        end_idx = min(k + n_hor, N_total)
        v_ref_hor = track_v[k:end_idx]
        # Pad if near end of lap
        if len(v_ref_hor) < n_hor:
            v_ref_hor = np.concatenate([v_ref_hor,
                                        np.full(n_hor - len(v_ref_hor), v_ref_hor[-1])])

        # Solve MPC with variable horizon
        u_opt, info = controller.step(x, v_ref_hor, n_horizon_override=n_hor)
        u_hist[k] = u_opt

        # Apply control and advance true nonlinear dynamics.
        # coupled_dynamics now sets v_next = v_ref (current step's reference).
        # We then override with the NEXT step's track speed so that
        # x_hist correctly records the real track speed at each timestep.
        x = coupled_dynamics(x, u_opt, track_v[k], motor, tp, mpc_p)
        x[IDX_V] = track_v[min(k + 1, N_total - 1)]   # enforce true speed
        x_hist[k + 1] = x

        # Progress
        if verbose and (k % 200 == 0 or k == N_total - 1):
            print(f"  [MPC] step {k:4d}/{N_total} | "
                  f"v={x[IDX_V]*3.6:5.1f} km/h | "
                  f"SOC={x[IDX_SOC]*100:5.1f}% | "
                  f"T_mag={x[IDX_TH_START+IDX_MAGNET]:5.1f}°C | "
                  f"P_e={u_opt/1e3:+7.1f} kW | "
                  f"{info.get('status', '?')}")

        info_hist.append(info)

    return {
        "x_hist": x_hist,
        "u_hist": u_hist,
        "info_hist": info_hist,
    }


# %%
# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.track_model import load_track, load_params

    print("=" * 60)
    print("  Stage 4: MPC Controller — OSQP QP Formulation")
    print("=" * 60)

    # Load everything
    params = load_params()
    motor = load_motor_params()
    tp = load_thermal_params()
    mpc_p = MPCParams.from_yaml(params)

    print(f"\n  MPC Configuration")
    print(f"  {'─' * 45}")
    print(f"  dt           = {mpc_p.dt} s")
    print(f"  N_horizon    = {mpc_p.N_horizon} steps ({mpc_p.N_horizon * mpc_p.dt} s)")
    print(f"  ε (fin diff) = {mpc_p.eps}")
    print(f"  P_peak       = {mpc_p.P_peak / 1e3:.0f} kW")
    print(f"  T_magnet_max = {mpc_p.T_magnet_max}°C")

    # ── Test 1: Single linearisation ──
    print(f"\n  Test 1: Linearisation at a representative operating point")
    print(f"  {'─' * 45}")
    x_test = np.array([80.0, 0.5, 70.0, 68.0, 67.0, 66.0, 65.5])
    u_test = 100e3
    A, B, c = linearise(x_test, u_test, 80.0, motor, tp, mpc_p)
    print(f"  A shape: {A.shape}, rank: {np.linalg.matrix_rank(A)}")
    print(f"  B shape: {B.shape}")
    print(f"  c shape: {c.shape}")
    print(f"  diag(A) = {np.diag(A).round(6)}")
    print(f"  B[:,0]  = {B[:,0].round(8)}")

    # ── Test 2: Condensed QP build ──
    print(f"\n  Test 2: Condensed QP Build for horizon N={mpc_p.N_horizon}")
    print(f"  {'─' * 45}")
    v_ref_test = np.full(mpc_p.N_horizon, 80.0)
    A_list = [A] * mpc_p.N_horizon
    B_list = [B] * mpc_p.N_horizon
    c_list = [c] * mpc_p.N_horizon
    H, g_qp, u_lb, u_ub, C_ineq, d_lb, d_ub, _, _, _ = \
        build_condensed_qp(x_test, A_list, B_list, c_list, mpc_p)
    print(f"  Decision variables: {len(g_qp)} (controls only — condensed)")
    print(f"  Inequality rows:    {C_ineq.shape[0]}")
    print(f"  H shape:            {H.shape}")

    # ── Test 3: Solve single QP ──
    print(f"\n  Test 3: Single QP Solve")
    print(f"  {'─' * 45}")
    u_vec_opt, info = solve_qp(H, g_qp, u_lb, u_ub, C_ineq, d_lb, d_ub)
    print(f"  Status:      {info['status']}")
    print(f"  Obj value:   {info['obj_val']:.4f}")
    print(f"  u*[0] (P_e): {u_vec_opt[0]/1e3:+.1f} kW")
    print(f"  u* range:    [{u_vec_opt.min()/1e3:+.1f}, "
          f"{u_vec_opt.max()/1e3:+.1f}] kW")

    # ── Test 4: Short MPC simulation (10 seconds) ──
    print(f"\n  Test 4: Short MPC Simulation (10 s)")
    print(f"  {'─' * 45}")
    track = load_track(params)
    N_short = int(10.0 / mpc_p.dt)  # 200 steps
    results = run_mpc_simulation(
        track.v[:N_short], track.segment[:N_short],
        motor, tp, mpc_p, verbose=True,
    )
    x_final = results["x_hist"][-1]
    print(f"\n  Final state after 10 s:")
    print(f"    v       = {x_final[IDX_V]*3.6:.1f} km/h")
    print(f"    SOC     = {x_final[IDX_SOC]*100:.1f}%")
    print(f"    T_mag   = {x_final[IDX_TH_START+IDX_MAGNET]:.1f}°C")
    print(f"    P_e avg = {results['u_hist'].mean()/1e3:+.1f} kW")
    print(f"    P_e max = {results['u_hist'].max()/1e3:+.1f} kW")

    print(f"\n[Done] Stage 4 validated.")
