# Evaluating trained policies

Activate the Isaac Gym environment used for training, then run:

```bash
conda activate trsppo
bash evaluation_metrics.sh --dry-run
bash evaluation_metrics.sh
```

The default matrix selects the latest saved PPO and TRS-PPO policies for Go2,
X1, and X1-Go2bend. Each policy is evaluated on **pronk, trot, pace, and bound**
at commanded forward velocities **-2, -1, -0.5, 0.5, 1, 2 m/s**, with zero lateral
and yaw commands. This gives 144 gait/speed cases and 14,400 trials by default.
Runs are selected by the modification time of `parameters.pkl` among runs with
the requested checkpoint. The dry run prints every resolved checkpoint and checks
its configuration, network dimensions, and deterministic action output on CPU.

Each case runs 100 parallel trials, with 250 settling steps followed by 1,000
measurement steps. At the current 50 Hz policy rate these are 5 and 20 seconds.
Gait frequency is 3 Hz, duty factor 0.5, foot swing height 0.08 m, stance width
0.25 m, and stance length 0.40 m. Body height, pitch, and roll commands are zero.
`--frequency`, `--velocities`, `--num-envs`, `--settle-steps`, and
`--measure-steps` override these settings consistently across selected policies.

Select individual policies or a smaller matrix with:

```bash
bash evaluation_metrics.sh --robot go2 --algorithm trsppo --dry-run
bash evaluation_metrics.sh --robot x1 --algorithm ppo --run RUN_NAME
bash evaluation_metrics.sh --robot x1_go2bend --algorithm trsppo \
  --run RUN_NAME --checkpoint 5999

# Short simulator smoke test; this is too short for meaningful gait statistics.
bash evaluation_metrics.sh --robot go2 --algorithm ppo --gait trot \
  --velocities 0.5 --num-envs 2 --settle-steps 10 --measure-steps 30
```

`--run` also accepts a full run directory. An explicit run requires one robot
and one algorithm. Checkpoints come from `checkpoints/ac_weights_last.pt` or
`checkpoints/ac_weights_NNNNNN.pt`; the matching saved `parameters.pkl` defines
the architecture, history length, controller, robot asset, and reference pose.
The launcher works from any directory. `--viewer` enables the simulator viewer;
otherwise evaluation is headless. Use `PYTHON=/path/to/python` to choose an
interpreter explicitly. `--device cpu` is available for simulator debugging.

Missing policies are reported before simulation; they are never silently skipped.
`deprecated_logs` is not searched automatically. Legacy 58D X1 policies are
trot-only and must be selected with `--gait trot`. Use newly trained four-gait
checkpoints for the full matrix.

## Gait selection and policy inputs

Evaluation locks the selected command vector, including after environment resets
and periodic command-resampling callbacks. The policy's first observation already
contains that command and its matching clocks. With `pacing_offset=False`:

| Physical gait | Phase | Offset | Bound | Synchronized feet |
| --- | --- | --- | --- | --- |
| Pronk | 0 | 0 | 0 | All four |
| Trot | 0.5 | 0 | 0 | FL/RR and FR/RL |
| Pace | 0 | 0 | 0.5 | FL/RL and FR/RR |
| Bound | 0 | 0.5 | 0 | FL/FR and RL/RR |

The mapping also accounts for saved `pacing_offset=True`. Evaluation labels use
physical foot timing: the existing training curriculum's pace/bound category
names are reversed when `pacing_offset=False`, but it still trains both patterns.
Runtime foot ordering is checked, and joint metrics use FL/FR/RL/RR order with
hip/thigh/calf joints in each leg.

The policy uses deterministic student actions (adaptation module followed by the
actor). PPO retains `HistoryWrapper`; TRS-PPO retains
`TimeReversalHistoryWrapper`, as in training. Every gait/speed case starts with
fresh histories and actuator memory. The saved control method, gains, action
scaling, and delay buffer remain in use. In this codebase,
`randomize_lag_timesteps` enables a fixed physics-step delay; it is preserved.

## Metric definitions and evaluation conditions

The evaluator includes an unchanged snapshot of the pure NumPy formulas from
`~/Documents/symm_rl_isaaclab/scripts/symm_locomotion/_leg_usage_metrics.py`,
with its BSD license in `go2_gym_learn/eval_metrics/LICENSE.isaaclab`. It runs
independently of that checkout. The source hash and metric thresholds are saved
in each evaluation manifest. Evaluation does not update policy weights or saved
training configurations.

Metrics include:

- Velocity RMSE, MAE, bias, relative error, gain, wrong-direction fraction,
  heading and lateral drift, directed progress, rise time, and settling time.
- Contact agreement, precision/recall/F1, duty-factor errors, touchdown/liftoff
  timing, synchronized-foot timing, cyclic ordering, and physical gait classification.
- Raw and effort-normalized squared torque, absolute mechanical work, vertical
  ground-reaction impulse, front/hind and left/right imbalance, load concentration,
  per-joint utilization and saturation, and foot impact loads.
- Separate velocity success, gait success, and joint velocity-and-gait success.

The reference's thresholds are preserved, including 20% relative forward-velocity
RMSE, 0.05 rad/s yaw RMSE, 80% gait agreement, correct gait family, two complete
cycles, and two touchdown/liftoff events per foot. All trials, including failures
during settling, remain in success-rate denominators. Each numeric mean has an
explicit sample count. Partial first episodes can contribute available measurements;
termination always fails the no-termination success condition. Metrics that cannot
be computed are omitted or null, never replaced with successful zero errors.

Measurements start after settling and are trimmed to complete common gait cycles,
matching the reference analysis. WTW stance occupies the first half-cycle, while
the reference metric convention uses the second half-cycle. Evaluation converts
the common phase by +0.5 cycles before computing contact agreement and events.

Evaluation uses a flat plane, no observation noise, no pushes or parameter
randomization, and zero initial yaw. It retains WTW's random initial joint
positions and base velocities, with the same case seed across algorithms/robots.
Self-collisions are disabled for all robots during evaluation; collision groups
isolate trials. With only the ground present, net foot forces therefore represent
ground contact rather than robot-to-robot or self contact. This setting is recorded
in the manifest. Ground friction is 1.0 and restitution is zero; robot material
properties retain their nominal asset values.

Terminal states are captured **before** the environment automatically resets.
Only each trial's first episode is analyzed, so reset jumps and subsequent
episodes cannot contaminate progress, power, or contact statistics. Episode timeouts
are extended beyond the requested window. Signals are sampled at policy frequency
after the final physics step, consistent with the reference's policy-rate traces;
they are not high-rate motor telemetry. Same formulas do not make the Isaac Gym
and Isaac Lab dynamics identical.

## Results

Outputs go to a new `evaluations/<UTC timestamp>/` directory, or a new directory
specified by `--output`:

```text
manifest.json
summary.csv
go2/ppo/policy.json
go2/ppo/velocity_+0.5/trot/trial_metrics.csv
go2/ppo/velocity_+0.5/trot/trial_metrics.json
go2/ppo/velocity_+0.5/trot/summary.json
go2/ppo/velocity_+0.5/trot/rollout.npz
... other robots, algorithms, velocities, and gaits ...
```

The policy metadata includes the checkpoint hash, actual robot mass, effort
limits, joint names, and controller settings. Trial JSON retains all per-foot and
per-joint arrays; CSV encodes array-valued fields as JSON. `summary.csv` compares
all completed cases and is updated after each case. The manifest reports running,
complete, or failed status. Existing output directories are never overwritten.

Raw NPZ traces contain time-major arrays for all parallel environments: body
velocity, position, heading, applied torque, joint power, vertical foot force,
gait phase, action clipping, and episode-end flags. Analysis stops at each trial's
first episode-end flag, even when its raw buffer also includes later episodes.
Use `--no-save-rollouts` to save metrics without these larger archives.
