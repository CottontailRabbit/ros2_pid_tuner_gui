# dg_pid_tuner_gui

PyQt5 standalone GUI for tuning `pid_controller/PidController` (ros2_controllers) gains on Tesollo grippers (3F_M / 3F_B / 4F / 5F / 5F_S).

## Features

- Connect to a running `ros2_control_node` by namespace (e.g. `/dg3f_m`).
- Live-edit per-joint gains: `p`, `i`, `d`, `i_clamp_max`, `i_clamp_min`.
- "Apply gains (live)" → sends `set_parameters` to the controller; the change takes effect immediately (standard ros2_control PidController behaviour).
- Run a step test on any joint and see overshoot / settling time / rise time / SSE / IAE.
- **Auto-tune** the selected joint with one of:
  - **Relay Feedback (Åström–Hägglund)** — safer auto Ziegler-Nichols using Tyreus-Luyben rules.
  - **Cohen-Coon** — single step, FOPDT fit, classic open-loop tuning.
  - **Bayesian Optimization (GP + EI)** — modern, sample-efficient global search. Uses [`scikit-optimize`](https://scikit-optimize.github.io) when available, otherwise a numpy fallback.
- Cost = `w₁·overshoot + w₂·settling + w₃·SSE` (weights configurable in UI).
- Save/Load `pid_controller.yaml` files in the standard ros2_control format.

## Build

```bash
cd <your_ros2_ws>   # workspace containing this package under src/
colcon build --packages-select dg_pid_tuner_gui
source install/setup.bash
```

System deps:

```bash
sudo apt install python3-pyqt5 python3-matplotlib python3-yaml
pip install --user scikit-optimize   # optional, enables faster Bayesian Optimization
```

## Run

```bash
# 1) bring up the gripper PID controller (separate terminal)
ros2 launch dg3f_m_driver dg3f_m_pid_controller.launch.py delto_ip:=169.254.186.72

# 2) launch the GUI
ros2 run dg_pid_tuner_gui dg_pid_tuner_gui
# or
ros2 launch dg_pid_tuner_gui dg_pid_tuner_gui.launch.py namespace:=/dg3f_m
```

## Workflow

1. Type the controller namespace (e.g. `/dg3f_m`) and click **Connect**.
2. The joint list is read from `dof_names`; the gain table is populated from the running controller via `get_parameters`.
3. Pick a joint and click **Run step test** to characterise the current PID's response.
4. Pick an algorithm (Bayesian Optimization recommended), set step Δ and capture window, then click **Auto-tune selected joint**.
5. The tuner repeatedly: applies candidate gains → publishes a step reference → captures `joint_states` for `duration` seconds → scores the response → proposes the next candidate. Best gains are auto-applied at the end and reflected in the table.
6. Click **Save YAML...** to persist the result to e.g. `dg3f_m_pid_controller.yaml`.

## Algorithm details

All FOPDT methods identify a first-order plus dead-time model `K e^{-Ls} / (τs+1)` from a single low-P step probe, then apply their respective tuning rule.

| Algorithm | Iterations | Notes |
| --- | --- | --- |
| Ziegler-Nichols (open-loop, FOPDT) | 2 | `Kp = 1.2 τ/(K L)`, `Ti = 2L`, `Td = 0.5L`. Aggressive — large overshoot is normal. |
| Cohen-Coon (FOPDT) | 2 | Better for plants with relatively large dead time (`L/τ` not tiny). Less overshoot than ZN. |
| Chien-Hrones-Reswick (CHR, 0% OS) | 2 | Setpoint regulation with no overshoot: `Kp = 0.6 τ/(K L)`, `Ti = τ`, `Td = 0.5L`. Conservative. Other modes available via `extra['chr_mode']`. |
| Internal Model Control (IMC) | 2 | `Kp = (2τ+L)/(K(2λ+L))`, `Ti = τ + L/2`, `Td = τL/(2τ+L)`. λ defaults to `max(0.1τ, 0.8L)` — smaller λ → faster but more aggressive. |
| Relay Feedback (Astrom-Hagglund) | ~5–10 | Probes increasing P-only gains until oscillation; first oscillation gives `Ku`, `Tu`. Tyreus-Luyben rule (less aggressive than classic ZN closed-loop). |
| Bayesian Optimization (GP+EI) | configurable (default 20) | Black-box optimisation. Models cost surface as a Gaussian Process and picks the next candidate via Expected Improvement. Recommended when the response is non-FOPDT (gear backlash, friction, saturation). |

## Auto-tune parameters (UI)

| Parameter | Meaning |
| --- | --- |
| **joint** | Which DOF to tune. Step test moves only this joint; others are held in place. |
| **algorithm** | Selects one of the methods above. |
| **step Δ from center [rad]** | Target step amplitude relative to the joint's `center` column. Direction alternates each trial to cancel drift. Clipped into the safe range. |
| **capture window [s]** | How long to record the response after each step. Should be ≥ a few settling times. |
| **settle at center [s]** | Delay holding the joint at `center` *before* each step, so each trial starts from the same state. |
| **iterations** | Total evaluations the autotuner performs. FOPDT methods need 2; relay 5–10; Bayesian 15–30. |
| **safety margin** | Fraction of the URDF joint range trimmed off each end. e.g. 0.20 → step is forced into `[lower+0.2·range, upper−0.2·range]`. Anything outside contributes a heavy cost penalty so the optimiser learns to avoid it. |
| **cost weights** | `cost = w_overshoot · overshoot/10 + w_settle · settling_time + w_sse · SSE/|Δ|`. Defaults: 1, 1, 5 (mild bias toward small SSE). |
| **gain search bounds** | Upper limits for the BO search (P, I, D). Lower bounds are 0/0.05. Lower → safer but underexplores; raise if BO converges at the boundary. |

### How divergence is prevented

* **center column**: every step starts from the joint's center (auto-filled from URDF, editable).
* **safety margin**: defines a "safe" sub-range strictly inside the URDF limits. Step targets are clipped into this band.
* **divergence penalty**: any sample outside the safe band contributes `≥ 50 + 50·excess/|Δ|` to that trial's cost. Bayesian Optimization's Gaussian Process therefore rapidly learns to stay away from gains that whip the joint into the limits.
* **pre-settle hold**: between every trial the joint is commanded to `center` and held for `settle_time` before the next step, so unstable gains can't compound.

## Safety notes

- Auto-tuning **moves the joint repeatedly**. Make sure the gripper has clearance.
- The step Δ is added to the joint's current position (not absolute); pick a value that stays inside the joint's safe range.
- Use the **Cancel** button to abort mid-run; the worker stops at the next iteration boundary.

## Compatibility

- ROS 2 Humble or newer.
- Requires `control_msgs/msg/MultiDOFCommand` for publishing the reference (Humble+ has it).
- Tested with `pid_controller/PidController` from `ros2_controllers`.
