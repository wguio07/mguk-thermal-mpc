# MGU-K Thermal-Constrained ERS Deployment Optimiser

**Ferrari F1 Engineering Academy 2026 — Wolfgang Guio**

A coupled thermal-electrical Model Predictive Controller (MPC) that simultaneously optimises MGU-K deployment strategy and manages permanent magnet demagnetisation risk over a real F1 race lap.

---

## Motivation

The 2026 F1 power unit regulations eliminate the MGU-H and increase MGU-K peak power from ~120 kW to ~350 kW. This makes the electrical system the dominant performance lever — but increased power density means rotor magnet temperatures that approach the NdFeB demagnetisation threshold (140 °C).

The central insight driving this project: **deployment strategy and thermal management are the same problem.** A high-power strategy that ignores magnet temperatures risks demagnetisation; over-conservative thermal management wastes available energy. Magnet temperature is also history-dependent — a high-current burst on one straight constrains your options 30 seconds later. This demands a controller that plans ahead.

MPC optimises over a 2-second prediction horizon, foresees thermal consequences, and adjusts power deployment proactively. This is fundamentally different from reactive thermal deration.

---

## System Architecture

This controller is a **sub-system optimiser** within a broader powertrain energy management stack. It assumes the ICE is running at a commanded operating point and is responsible solely for optimal electrical power deployment with hard thermal guarantees.

```
FastF1 Lap Data (Monza 2025)
        │
        ▼
┌───────────────┐     state: [v, SOC, T_winding, T_magnet,   ┌──────────────────┐
│  track_model  │ ──► T_stator, T_housing, T_coolant]       │  mpc_controller  │
└───────────────┘                                     ──────►│  (OSQP, RTI)     │
                                                             └────────┬─────────┘
┌───────────────┐                                                     │ P_e (W)
│  pmsm_losses  │ ◄───────────────────────────────────────────────────┘
│  (Cu + Fe)    │
└───────┬───────┘
        │ P_loss
        ▼
┌───────────────┐
│thermal_network│ → T_magnet < 140 °C (hard constraint)
│  (5-node ODE) │
└───────────────┘
```

At each 0.05 s timestep: the MPC solver receives the current state, computes optimal `P_e`, the loss model evaluates electrical losses at that power level, the thermal ODE updates all node temperatures, and the updated state feeds back into the next MPC iteration. This closes the feedback loop continuously.

---

## Physics

### PMSM Loss Models (`src/pmsm_losses.py`)

| Loss type | Equation |
|-----------|----------|
| Copper (resistive) | P_cu = (3/2) × R_s × I_s² |
| Iron (Steinmetz) | P_fe = k_h·f·B_peak^α + k_e·(f·B_peak)² |

### 5-Node Thermal Network (`src/thermal_network.py`)

Lumped capacitance model with linear thermal resistances:

```
C_i · dT_i/dt = P_loss,i + Σ (T_j − T_i) / R_ij
```

| Node | C (J/K) |
|------|---------|
| Windings | 800 |
| Stator iron | 2000 |
| Rotor / magnets | 400 |
| Housing | 5000 |
| Coolant | 3000 |

Coolant inlet fixed at 65 °C (constant, from main radiator circuit).

### MPC Formulation (`src/mpc_controller.py`)

| Parameter | Value |
|-----------|-------|
| Solver | OSQP (convex QP) |
| Prediction horizon | 2 s (40 steps) |
| Timestep Δt | 0.05 s |
| Linearisation | Numerical Jacobians recomputed every step (RTI) |
| Warm-starting | Previous solution shifted forward as initial guess |

**Cost function:**

```
min  −w_v·v  +  w_u·P_e²  +  w_soc·(SOC − SOC_ref)²
```

Speed reward on straights, control effort penalty, SOC tracking to preserve energy reserve.

**Constraints:**

| Constraint | Bound |
|------------|-------|
| SOC | [20 %, 95 %] |
| T_magnet (hard) | < 140 °C |
| Peak power | ±350 kW |
| Continuous power | ±120 kW (time-averaged) |
| Total lap energy | ≤ 4 MJ (2026 regulations) |

---

## Results — Monza 2025 Lap (84 s)

| Metric | Value |
|--------|-------|
| MPC iterations | 1,680 |
| Solver convergence | 100 % |
| Mean solve time | 12 ms/step (warm-start) |
| Peak magnet temperature | **134.4 °C** |
| Safety margin vs. 140 °C limit | **5.6 °C** |
| All state/control constraints | ✅ Satisfied |

Without the quadratic thermal barrier penalty (tuned 10 °C below T_max), the controller drove magnet temperature to 180 °C — a demagnetisation failure. With the barrier, it holds safely within the safe zone while maximising deployment.

---

## Repository Structure

```
mguk-thermal-mpc/
├── src/
│   ├── track_model.py       # FastF1 lap extraction; Monza profile + synthetic fallback
│   ├── pmsm_losses.py       # Copper (I²R) and iron (Steinmetz) loss models
│   ├── thermal_network.py   # 5-node ODE thermal network
│   ├── mpc_controller.py    # OSQP QP formulation, warm-starting, RTI Jacobians
│   └── __init__.py
├── params/                  # Motor and vehicle parameters
├── plots/                   # Output: deployment profile, temperature vs limit, SOC
├── main.py                  # Full lap coupling — runs 1,680 MPC solves
├── requirements.txt
└── setup_env.bat            # Windows environment setup
```

---

## Motor & Vehicle Parameters

| Parameter | Value |
|-----------|-------|
| Stator resistance R_s | 2 mΩ/phase |
| Magnet flux ψ_m | 0.08 Wb |
| Pole pairs p | 4 |
| Vehicle mass | 800 kg (car + driver) |
| Drag C_d·A | 0.7 m² (Monza low-drag) |
| Track data | Monza 2025 via FastF1 (synthetic fallback: 84 s, 5793 m, v_max = 350 km/h) |

---

## Installation & Usage

**Requirements:** Python 3.9+, Windows (for `setup_env.bat`) or any platform with pip.

```bash
# Clone the repository
git clone https://github.com/wguio07/mguk-thermal-mpc.git
cd mguk-thermal-mpc

# Windows — automated environment setup
setup_env.bat

# Or manually
pip install -r requirements.txt

# Run the full Monza lap simulation
python main.py
```

Output plots are saved to `plots/`:
- MGU-K deployment profile vs track position
- Magnet temperature trajectory vs 140 °C demagnetisation limit
- SOC trajectory with constraint bounds

---

## Key Assumptions & Limitations

**Assumptions:**
- Motor parameters sourced from published literature for 300–400 kW class automotive PMSMs, not fitted to specific F1 hardware
- Coolant inlet temperature held constant at 65 °C; real systems have transient coolant response
- Steinmetz coefficients typical for M330-35A silicon steel
- Point-mass vehicle dynamics; no suspension, tyre, or driver lag modelling

**Limitations:**
- Single-lap validation; no multi-lap thermal soak or degradation analysis
- RTI-MPC recomputes Jacobians every step but cannot guarantee global optimality under steep nonlinearities
- Inverter losses, transient current limits, and motor saturation are not modelled
- ±10 % variation in motor resistance, coolant conductance, or magnet thermal capacity shifts peak temperature by ~2–3 °C
- 2 s prediction horizon is shorter than full braking zones; longer horizons improve foresight at higher compute cost

---

## Regulatory Context

Designed against the 2026 FIA F1 Power Unit Technical Regulations:
- MGU-H eliminated
- MGU-K peak power: ~350 kW (up from ~120 kW)
- Maximum deployable energy per lap: 4 MJ
- Hard demagnetisation limit for NdFeB permanent magnets: 140 °C

---

## Author

**Wolfgang Guio** — MSc Motorsport Engineering (Distinction), Oxford Brookes University  
[GitHub](https://github.com/wguio07) · [LinkedIn](https://www.linkedin.com/in/wolfgangguio)
