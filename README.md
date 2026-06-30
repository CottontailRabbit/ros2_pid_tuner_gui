# ros2_pid_tuner_gui

PyQt5 standalone GUI for tuning **ros2_control** PID gains on any controller that
exposes runtime-reconfigurable gains. The controller type is **auto-detected**:

| Controller | DOF parameter | Gain fields | Command interface |
| --- | --- | --- | --- |
| `pid_controller/PidController` | `dof_names` | `p, i, d, i_clamp_max, i_clamp_min` | `control_msgs/MultiDOFCommand` → `<ctrl>/reference` |
| `joint_trajectory_controller/JointTrajectoryController` | `joints` | `p, i, d, i_clamp, ff_velocity_scale` | `trajectory_msgs/JointTrajectory` → `<ctrl>/joint_trajectory` |

The gain table columns, parameter paths, and step-command message are all selected
automatically from whichever controller you connect to.

## Features

- Connect to a running `ros2_control_node` by namespace and controller name; the
  controller type (pid_controller / joint_trajectory_controller) is detected from
  its parameters.
- Live-edit per-joint gains and **Apply gains (live)** → `set_parameters`, effective
  immediately.
- Run a step test on any joint and see overshoot / settling time / rise time / SSE / IAE.
- **Auto-tune** the selected joint with one of: Ziegler-Nichols, Cohen-Coon,
  Chien-Hrones-Reswick, Internal Model Control, Relay Feedback, or Bayesian
  Optimization (GP + EI).
- Cost = `w₁·overshoot + w₂·settling + w₃·SSE` (weights configurable in UI).
- Save/Load controller gain YAML files in the standard ros2_control format
  (handles both `dof_names`- and `joints`-style controllers).

## Build

```bash
cd <your_ros2_ws>   # workspace containing this package under src/
colcon build --packages-select ros2_pid_tuner_gui
source install/setup.bash
```

System deps:

```bash
sudo apt install python3-pyqt5 python3-matplotlib python3-yaml
pip install --user scikit-optimize   # optional, enables faster Bayesian Optimization
```

## Run

```bash
ros2 run ros2_pid_tuner_gui ros2_pid_tuner_gui
# or
ros2 launch ros2_pid_tuner_gui ros2_pid_tuner_gui.launch.py \
    namespace:=/your_ns controller:=pid_controller
```

For a joint_trajectory_controller:

```bash
ros2 launch ros2_pid_tuner_gui ros2_pid_tuner_gui.launch.py \
    namespace:=/your_ns controller:=joint_trajectory_controller
```

## Workflow

1. Type the controller **namespace** (empty if none) and **controller** name, then
   click **Connect**. The type is auto-detected and logged.
2. The joint list is read from `dof_names`/`joints`; the gain table columns and values
   are populated from the running controller via `get_parameters`.
3. Pick a joint and click **Run step test** to characterise the current response.
4. Pick an algorithm (Bayesian Optimization recommended), set step Δ and capture
   window, then click **Auto-tune selected joint**.
5. The tuner repeatedly: applies candidate gains → commands a step reference →
   captures `joint_states` for `duration` seconds → scores the response → proposes the
   next candidate. Best gains are auto-applied at the end and reflected in the table.
6. Click **Save YAML...** to persist the result.

> **JTC note:** the step command is a single-waypoint `JointTrajectory` published on
> `<ctrl>/joint_trajectory` (topic interface), which the controller tracks with its
> internal PID. Make sure the controller is active and accepting trajectories.

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

- Auto-tuning **moves the joint repeatedly**. Make sure the mechanism has clearance.
- The step Δ is relative to the joint's `center` (not absolute); pick a value that stays inside the joint's safe range.
- Use the **Cancel** button to abort mid-run; the worker stops at the next iteration boundary.

## Compatibility

- ROS 2 Humble or newer.
- `pid_controller` requires `control_msgs/msg/MultiDOFCommand` (Humble+ has it).
- `joint_trajectory_controller` uses the `joint_trajectory` topic interface and
  `trajectory_msgs`.
- Tested against `ros2_controllers`' `pid_controller/PidController` and
  `joint_trajectory_controller/JointTrajectoryController`.
