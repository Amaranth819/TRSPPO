# Training with time-reversal PPO

Select the algorithm using the existing training entry point:

```bash
# Original PPO (also the default when --algorithm is omitted)
python scripts/train.py --algorithm ppo --headless

# Time-reversal symmetry PPO (TRS-PPO)
python scripts/train.py --algorithm trsppo --headless
```

Run these commands in the repository's Isaac Gym environment. `python scripts/train.py --help`
works without importing Isaac Gym. Omit `--headless` to display the viewer.

## Configuration

Both algorithms use the existing MLP actor, adaptation network, privileged critic,
12 joint-position actions, and friction/restitution privileged inputs. Go2, X1,
and `x1_go2bend` share 70D observations with 20 history frames (1400 inputs), including
all 15 commands. Every robot trains on pronk, trot, pace, and bound using the same
command ranges, curricula, rewards, termination thresholds, terrain, and domain
randomization. The configured gait-frequency range is 2–4 Hz, with 50% gait duration.
X1 and `x1_go2bend` use position PD gains `Kp=30`, `Kd=1.2`; Go2 uses its actuator
network. The X1 variants differ only in the four rear thigh/calf
reference angles: X1 uses `-0.7/1.5`, while `x1_go2bend` uses `0.7/-1.5`, matching
the front legs. Both share the X1 URDF/meshes and joint limits.
TR-PPO selects a separate PPO subclass and a reset-safe history wrapper. The
original `ppo.py`, `actor_critic.py`, `rollout_storage.py`, and standard history
wrapper retain their existing behavior.

| Option | Default |
| --- | --- |
| `--algorithm` | `ppo` |
| `--robot` | `go2`; also supports `x1` and `x1_go2bend` |
| `--num-steps-per-env` | 24 for both algorithms and all robots |
| `--num-envs` | 4096 |
| `--iterations` | 10000 |
| `--tr-actor-loss-coef` | 0.1 |
| `--tr-value-loss-coef` | 0.0 (disabled) |
| `--tr-warmup-iterations` | 0 PPO updates |
| `--tr-min-abs-command-velocity` | 0.0 m/s, including standing commands |

TR-PPO requires at least `history_length + 2` rollout steps (22 for Go2, X1,
and `x1_go2bend`). The extra step supplies the next action needed by the second
previous-action field. Longer rollouts consume more GPU memory. Pair histories are
constructed per mini-batch rather than duplicating the entire history buffer.

For a short simulator smoke run with the TR losses active immediately:

```bash
python scripts/train.py --algorithm trsppo --headless \
    --num-envs 64 --iterations 2 --tr-warmup-iterations 0
```

For a controlled PPO/TR-PPO comparison, use the same environment count, rollout
length (`--num-steps-per-env 24` for every robot), and training budget in both runs.
Setting both TR coefficients to zero delegates updates to the original PPO, but
the TR selection still uses its reset-safe history wrapper and rollout settings.

## Reversal and losses

The implementation follows the temporal actor/value consistency approach in
`symm_rl_isaaclab/.../config/symm_quadruped/time_reversal_ppo.py`, adapted to this
repository's observation layout and policy interfaces.

For a history ending at time `t`, TR-PPO reverses frame order, negates commanded
velocities and joint velocities, and maps gait phase/offset/bound commands to their
negatives modulo one. Gravity, joint positions, non-directional commands, and the
position-action signs are preserved. At the reversed counterpart of state `s_j`,
the two previous-action slots contain `a_j` and `a_(j+1)`. Their values use the
environment's clipping; PPO log probabilities retain the original sampled actions.

The current implementation validates the 70D/58D layouts, phase-command scale 1.0,
and fixed 50% gait duration. The phase reflection `phi' = (0.5 - phi) mod 1`
preserves the sine clock value at each paired frame and reverses its temporal
evolution. Other observation layouts or duty factors need their own reversal rules.

The shared per-frame mapping, applied after reversing the oldest-first frame order, is:

| Python slice | Observation | Mapping |
| --- | --- | --- |
| `0:3` | Projected gravity | Preserve |
| `3:6` | Linear/yaw velocity commands | Negate |
| `6:8` | Body height and gait frequency commands | Preserve |
| `8:11` | Gait phase, offset, and bound commands | Negate modulo one |
| `11:18` | Remaining commands | Preserve |
| `18:30` | Joint positions | Preserve |
| `30:42` | Joint velocities | Negate |
| `42:54` | Previous action | Replace with outgoing action `a_j` |
| `54:66` | Second previous action | Replace with `a_(j+1)` |
| `66:70` | Sine clocks | Preserve under the validated 50% gait duration |

Legacy X1 trot-only checkpoints observe only the three velocity commands, in
`[vx, vy, yaw]` order; their other 12 commands still control the environment and rewards.
Their 58D frame contains gravity
at `0:3`, velocity commands at `3:6`, joint positions at `6:18`, joint velocities
at `18:30`, previous actions at `30:42` and `42:54`, and sine clocks at `54:58`.
Reversal negates the visible commands and velocities, shifts both action histories,
and preserves positions, gravity, and clocks. Pair masks compare the visible commands.

Reversed histories at `t` are paired with reference histories at `t - H`, where
`H` is the history length, before shuffling. Pairs are excluded if they cross a
reset (including the boundary supplying the next action), have padded histories,
contain command changes, or have differing privileged parameters. Reference
network outputs are evaluated with gradients disabled.

By default, the training loss adds only actor-mean consistency, with weight 0.1,
to the usual PPO loss. Value consistency has weight 0; the normal PPO critic loss
still trains the critic. PPO surrogate, critic, entropy, and supervised adaptation
losses and their coefficients are unchanged, and no additional latent loss or
actor-mean bound penalty is introduced. Value consistency remains available only
when explicitly enabled with `--tr-value-loss-coef`.

The adapter is evaluated separately on the reversed and reference histories.
Its outputs estimate friction and restitution and enter the actor without sign
changes or permutations. Their numerical estimates need not match between histories.
Actor TR gradients reach both the actor body and adaptation network through the
reversed history; the entire reference branch is evaluated without gradients.
The existing supervised adaptation update is retained. This is a learned
consistency regularizer, not a claim that the damped/contact simulator is exactly
time-reversible. Reversed samples are not assigned copied PPO advantages or returns.

## Metrics, checkpoints, and tests

Training saves runs under
`runs/gait-conditioned-agility/{robot}/{algorithm}/{date_time}/`, where `robot`
defaults to `go2` and `algorithm` is `ppo` or `trsppo`. The UTC timestamp combines
date and time as `YYYY-MM-DD_HH-MM-SS.ffffff`. `--robot <name>` selects the model,
training settings, and output directory. `train.sh` accepts `bash train.sh --robot x1`
or `bash train.sh --robot x1_go2bend`; `ROBOT=<name> bash train.sh` also works.
Checkpoints, `metrics.pkl`, and the
default `tensorboard/` directory all live within each run. TR runs add these metrics/charts:
`tr_actor_loss`, `tr_value_loss`, `tr_valid_pair_fraction`, and `tr_active`.
Warmup uses the original PPO update and records zero TR losses. A zero valid-pair
fraction after warmup means no symmetry gradient is being applied.

The policy weights and `body_latest.jit` / `adaptation_module_latest.jit` retain
their existing formats. New runs for every robot use 1400 history inputs. Earlier
X1 and `x1_go2bend` checkpoints used a trot-only task and 58D observations, with
1160 inputs for 20 history frames or 1740 for 30. Load their saved configurations
for inference; start new runs for the shared 70D, four-gait task. Older Go2
checkpoints with 30-frame histories require their original 2100-input configuration.
Checkpoints require matching observation layout, history length, and robot configuration.
TR checkpoints also save `time_reversal_state_*.pt` and
`time_reversal_state_last.pt` with optimizer state, configuration, and update count.
The runner's existing `RunnerArgs.resume` / `resume_path` mechanism loads this
sidecar for a TR run; it requires matching TR configuration and a TR checkpoint.
The existing remote checkpoint loader is retained; no new resume CLI is introduced.

## Viewing training reward curves

Training writes `metrics.pkl` in each run directory using `ml_logger`. From the
repository root, plot the latest PPO and TRS-PPO runs together with:

```bash
python scripts/plot_training.py
```

Use `--robot <name>` to select a different robot's latest PPO/TRS-PPO runs.
For Go2, the plotter also reads the previous directory layout when no new run is available.

Open `tmp/training_rewards.png` to view the total reward against training iteration.
Rerun the command to refresh it during training. The bold lines use a trailing
average over 11 logged points; faint lines show the raw values. Use `--smooth 1`
to disable smoothing. This reads existing logs and does not restart training.

To plot a different reward component, for example:

```bash
python scripts/plot_training.py --metric train/episode/rew_tracking_lin_vel/mean \
    --output tmp/tracking_reward.png
```

Use `--runs <run-directory> [<another-run-directory> ...]` to select runs explicitly.

Run the CPU tests using an environment with the project's Python dependencies,
PyTorch, and pytest:

```bash
python -m pytest -q tests
```

Tests cover action alignment, reversal consistency, clocks, boundary/command masks,
gradient flow, finite PPO updates, warmup and disabled-loss equivalence to PPO,
history resets, algorithm selection, checkpoint state, and JIT export. They do not
require launching Isaac Gym.
