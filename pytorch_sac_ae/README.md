# SAC+AE implementation in PyTorch

This is PyTorch implementation of SAC+AE from

**Improving Sample Efficiency in Model-Free Reinforcement Learning from Images** by

[Denis Yarats](https://cs.nyu.edu/~dy1042/), [Amy Zhang](https://mila.quebec/en/person/amy-zhang/), [Ilya Kostrikov](https://github.com/ikostrikov), [Brandon Amos](http://bamos.github.io/), [Joelle Pineau](https://www.cs.mcgill.ca/~jpineau/), [Rob Fergus](https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php).

[[Paper]](https://arxiv.org/abs/1910.01741) [[Webpage]](https://sites.google.com/view/sac-ae/home)

## Citation
If you use this repo in your research, please consider citing the paper as follows
```
@article{yarats2019improving,
    title={Improving Sample Efficiency in Model-Free Reinforcement Learning from Images},
    author={Denis Yarats and Amy Zhang and Ilya Kostrikov and Brandon Amos and Joelle Pineau and Rob Fergus},
    year={2019},
    eprint={1910.01741},
    archivePrefix={arXiv}
}
```

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with:
```
source activate pytorch_sac_ae
```

## Instructions
To train an SAC+AE agent on the `cheetah run` task from image-based observations  run:
```
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --seed 1
```
This stores train/eval logs, TensorBoard events, and evaluation videos under
`logs/cheetah_run/sac_ae/1/`. Monitor training by running:
```
tensorboard --logdir logs
```
and opening up tensorboad in your browser.

The console output is also available in a form:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
a training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if is trained from pixels and decoder)
```
while an evaluation entry:
```
| eval | S: 0 | ER: 21.1676
```
which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10).

### JSON configuration

Use `--cfg` to load a full or partial argument JSON file. Settings are applied
in this order: built-in defaults, JSON values, then explicit command-line
arguments (regardless of where `--cfg` appears). For example, from `pytorch_sac_ae/`:

```bash
python -u train.py --cfg args/cheetah/bidirectional/trsppo.json \
    --seed 2 --no_save_video
```

Without `--cfg`, the existing defaults apply. JSON keys use argument names without
`--`, such as `actor_lr` and `save_video`, with JSON numbers and booleans; unknown
keys and invalid types are rejected. Use `--save_*` / `--no_save_*` for output
flags, `--gym_env` / `--no_gym_env`, `--prioritized_replay` /
`--no_prioritized_replay`, and `--time_rev true` / `--time_rev false` to override
boolean settings. The run's `work_dir/args.json` records the final merged settings
and can itself be reused with `--cfg`.

`work_dir` is always computed as `logs/{domain}_{task}/{agent}/{seed}/` from the
final merged arguments, relative to the current working directory. Stored
`work_dir` values in JSON files are ignored, and there is no `--work_dir` option.
The example above writes to `logs/cheetah_bidirectional/trsppo/2/`.

### Submit multiple jobs with HTCondor

Edit the `queue agent, cfg, seed from (...)` table in `condor_learn.sub`. Each row
starts one job with its own `--agent`, `--cfg`, and `--seed`, for example:

```text
queue agent, cfg, seed from (
ppo     args/quadruped/walk/ppo.json       1
trsppo  args/quadruped/walk/trsppo.json    1
ppo     args/cartpole/swingup/ppo.json    1
trsppo  args/cartpole/swingup/trsppo.json 1
)
```

From the repository root, validate the submission and then submit it:

```bash
cd pytorch_sac_ae
mkdir -p condor_logs
condor_submit -dry-run /tmp/trsppo-condor.ads condor_learn.sub
condor_submit condor_learn.sub
```

The [HTCondor dry run](https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html)
writes the expanded job descriptions without submitting jobs. The wrapper
activates the `tsda` Conda environment and runs
`python -u train.py --agent <agent> --cfg <cfg> --seed <seed> --no_save_video`. Config paths are
relative to `pytorch_sac_ae/`; the explicit agent and seed override their JSON settings.
`--no_save_video` disables video recording even if the JSON sets `save_video` to
`true`. Use the same flag when running `train.py` directly.
The wrapper also adds the environment's libraries to `LD_LIBRARY_PATH` so
Pillow/torchvision can use its C++ runtime.

Each job requests one CPU and 8 GB of RAM; uncomment `request_gpus = 1` in the
submit file to request a GPU. This setup uses a shared filesystem: worker nodes must be able
to access this repository, `/home/<user>/miniconda3`, and the MuJoCo installation.
Condor logs use `condor_logs/condor_learn.<cluster>.<process>.{out,err,log}`.
Create `condor_logs/` before submitting; a missing directory causes jobs to be held.
Training results
still use `logs/{domain}_{task}/{agent}/{seed}/`; jobs with the same domain, task,
agent, and seed share that directory. For repeated runs, add rows with different
seed values so each run has its own output.

## PPO with state observations

Select `--agent ppo` to train separate actor and state-value MLPs, each with two
`hidden_dim` hidden layers and tanh activations. The Gaussian actor has learned,
state-independent log standard deviations and tanh-squashed actions in [-1, 1].
There is no encoder or decoder. SAC remains the default agent.

From `pytorch_sac_ae/` in the `tsda` environment:

```bash
python -u train.py --agent ppo \
    --domain_name cheetah --task_name run --action_repeat 4 \
    --num_train_steps 1000000 --seed 1 \
    --save_tb --save_video --save_model
```

The implementation follows [PPO's clipped policy objective](https://arxiv.org/abs/1707.06347)
with [generalized advantage estimation](https://arxiv.org/abs/1506.02438).
Each rollout uses the current policy; old log probabilities, values and returns
stay fixed during shuffled minibatch optimization. Advantages are normalized
across the rollout. The critic minimizes squared error to the GAE return.
Timeouts bootstrap from the final observation, while advantages stop at every
episode boundary. The final partial rollout is also trained.

Shared arguments include `hidden_dim`, `batch_size`, `discount`, `actor_lr`,
`critic_lr`, `actor_beta`, `critic_beta`, the actor log-standard-deviation bounds,
`num_train_steps`, and evaluation/output settings. Current shared defaults use
256 hidden units, batch size 256, discount 0.99 and learning rates 0.001.

| PPO argument | Default | Meaning |
| --- | --- | --- |
| `--rollout_steps` | 1000 | Agent steps collected before updating |
| `--ppo_epochs` | 10 | Maximum passes over each rollout |
| `--gae_lambda` | 0.95 | GAE trace decay |
| `--ppo_clip` | 0.2 | Policy probability-ratio clipping range |
| `--entropy_coef` | 0.0 | Squashed-policy entropy bonus weight |
| `--value_coef` | 0.5 | Critic MSE weight |
| `--max_grad_norm` | 0.5 | Gradient norm limit for each network |
| `--target_kl` | 0.03 | Stop updates when minibatch approximate KL exceeds this; 0 disables |

PPO does not use SAC's `init_steps`, replay capacity, target-network settings,
update-frequency settings, or temperature settings. Pixel encoders/decoders,
prioritized replay, SAC replay augmentation (`--time_rev`), and `gym_env` are
rejected by the PPO training path. Training starts with policy samples, without random warmup.

`num_train_steps`, `rollout_steps` and logged steps count agent actions. With
`action_repeat=4`, each action sums up to four environment rewards. Evaluation
uses deterministic actions in a separate environment seeded with `seed + 10000`,
at step 0, every `eval_freq` steps, and at the final training step.

Outputs under `work_dir` (`logs/{domain}_{task}/{agent}/{seed}/`):

- `args.json`: run configuration.
- `train.log`: JSON lines containing completed episode returns/lengths and PPO
  update metrics (policy/value losses, entropy, KL, clipping fraction, explained
  variance, number of minibatch updates and rollout size). Episode and update
  records are separate; an unfinished final episode is not logged as complete.
- `eval.log`: mean return over `num_eval_episodes` at each evaluation.
- `tb/`: TensorBoard scalar events when `save_tb` is enabled.
- `model/ppo_<step>.pt`: actor, critic and optimizer states at evaluation
  checkpoints and at training completion when `save_model` is enabled.
- `video/<step>.mp4`: the first evaluation episode when `save_video` is enabled.
- `buffer/rollout.npz`: latest completed rollout when `save_buffer` is enabled;
  this is an inspection snapshot, not replay data.

The existing defaults enable all four optional outputs. Disable any with
`--no_save_tb`, `--no_save_model`, `--no_save_video`, or `--no_save_buffer`.
Different seeds get separate directories. Repeating the same domain, task,
agent, and seed reuses its directory: the existing logger replaces previous
train/evaluation logs and TensorBoard events there.

Checkpoints can be loaded with `PPOAgent.load(model_dir, step)` into an agent
constructed with the same network settings. This restores networks and
optimizers; it does not resume environment state or an in-progress rollout.

To run the focused PPO/TRSPPO tests from the repository root:

```bash
python -m unittest discover -s pytorch_sac_ae -p test_trsppo.py -v
```

## Plotting reward curves

From `pytorch_sac_ae/`, plot PPO and TRSPPO evaluation returns over training:

```bash
python -m pip install matplotlib tensorboard
python plot_rewards.py --domain cheetah --task bidirectional \
    --step_interval 50000
```

The script reads `eval/episode_reward` from all runs under
`logs/{domain}_{task}/{ppo,trsppo}/{run}/tb/`. Use `--log_root` to change the
default `logs/` root. It saves `plots/{domain}_{task}.png`; use
`--output path/to/figure.pdf` (or `.png`) to choose another destination.
PPO is always blue and TRSPPO orange. Solid lines show the mean and shaded
bands show +/- one population standard deviation **across runs**, not across
evaluation episodes. A single run has zero standard deviation.

`--step_interval X` samples at steps `0, X, 2X, ...` (default: 10000), without
smoothing. Logged values are used directly at matching steps; missing grid
points are linearly interpolated between neighboring evaluations. Runs only
contribute within their recorded step range, so shorter runs are never extended
and the number contributing can decrease later in training. The script prints
the included run paths, their step ranges, and contributing-run counts. Every
available run is included, even if seeds or hyperparameters differ.

Use `--max_training_steps N` to plot only training steps strictly below `N`.
Omit it to plot the full training curve. To apply the limit to all plots in the
shell script, run `bash plot_rewards.sh --max_training_steps 1000000`.

## TRSPPO

`--agent trsppo` adds time-reversal actor consistency to the existing PPO update.
The optional `--tr_value_coef` enables a separate critic symmetry ablation.
The actor and critic use the same two-layer MLPs as vanilla PPO and take one
state observation per action. Supported tasks and flattened observation mappings:

| Domain / task | Observation dim | Action dim | Preserved features | Negated velocities |
| --- | --- | --- | --- | --- |
| `cheetah/run`, `cheetah/bidirectional` | 17 | 6 | `0:8` positions | `8:17` |
| `acrobot/swingup` | 6 | 1 | `0:4` arm orientations | `4:6` |
| `cartpole/swingup` | 5 | 1 | `0` cart position, `1:3` pole cosine/sine | `3:5` |
| `reacher/hard` | 6 | 2 | `0:2` joint angles, `2:4` target displacement | `4:6` |
| `pendulum/two_poles` | 6 | 2 | `0:4` link cosine/sine orientations | `4:6` |
| `pendulum/three_poles` | 9 | 3 | `0:6` link cosine/sine orientations | `6:9` |
| `finger/spin` | 9 | 2 | `0:4` finger angles and spinner tip position, `7:9` touch readings | `4:7` |
| `walker/stand`, `walker/walk`, `walker/run` | 24 | 6 | `0:14` body orientations, `14` torso height | `15:24` |
| `hopper/stand`, `hopper/hop` | 15 | 4 | `0:6` positions, `13:15` touch readings | `6:13` |
| `quadruped/walk` | 78 | 12 | `0:16` joint positions, `32:44` activations, `47` uprightness, `48:51` accelerometer, `54:78` force/torque | `16:32` joint, `44:47` torso linear, `51:54` gyro |

Slices use Python's zero-based, end-exclusive convention. All controls
retain their signs and ordering. Orientation sine/cosine features remain
unchanged; walker does not swap legs. The agent selects a mapping by
`domain_name` and `task_name` and validates observation/action dimensions.
The custom pendulum tasks have a motor at every joint. Their different numbers
of links require an explicit `task_name` in `TRSPPOAgent`, `reverse_observation`,
`reverse_action`, and `get_time_reversal_spec`; the training CLI passes it
automatically. Quadruped also requires an explicit `task_name='walk'`; its other
tasks are not registered. Existing domain-only calls remain supported for the
other domains.

Cartpole's single action applies force to the cart and is unchanged by the
mapping. Only the single-pole `swingup` task is currently registered for TRSPPO;
multi-pole tasks have different observation dimensions and are rejected.

Reacher hard uses two direct torque controls. Its target displacement depends
only on positions and is preserved. The sparse reward depends only on fingertip
distance to the target, so it is invariant to velocity reversal. Joint damping
and wrist joint-limit dynamics prevent exact trajectory reversal.

Hopper's actions control waist, hip, knee and ankle torques. Its toe and heel
touch features are log-transformed force readings. Preserving them is an
approximation: recomputing contact forces after reversing velocities can change
these readings. The mapping does not claim exact reversal of the full simulated
observation or dynamics.

Finger spin controls the proximal and distal joint torques. Its three velocities
include both finger joints and the spinner hinge, all of which are negated.
The two log-transformed touch readings are preserved with the same approximation
as hopper's sensors. The spinner tip position is preserved, not reflected.

Quadruped walk reverses three separate velocity ranges. Its accelerometer and
toe force/torque readings are preserved as an approximation: recomputing these
sensors after reversing velocities can change them because of damping and
contacts. Its 12 actions are filtered position targets, with activation states
at `32:44`. Preserving these states and actions is an actor-regularization
assumption, not exact reversal of the actuator filters. For the continuous-time
filter `dw/dt = (u - w) / 0.1`, preserving `w` while reversing its derivative
would require `u_reversed = 2*w - u` in native control units. This implementation
uses identity actions and does not apply that state-dependent correction.

```bash
python -u train.py --agent trsppo \
    --domain_name cheetah --task_name bidirectional --action_repeat 4 \
    --tr_policy_coef 0.1 --num_train_steps 1000000 --seed 1 \
    --save_tb --save_video
```

To use the other supported tasks, set `--domain_name acrobot --task_name swingup`,
`--domain_name cartpole --task_name swingup`,
`--domain_name reacher --task_name hard`,
`--domain_name pendulum` with `--task_name two_poles` or `three_poles`,
`--domain_name finger --task_name spin`,
`--domain_name quadruped --task_name walk`,
`--domain_name walker` with `--task_name stand`, `walk`, or `run`, or
`--domain_name hopper` with `--task_name stand` or `hop`. The output directory
follows the selected domain, task, agent, and seed automatically. The same
PPO/TRSPPO hyperparameters and logging options apply.

TRSPPO runs write to
`logs/{domain}_{task}/trsppo_policy_{tr_policy_coef}_value_{tr_value_coef}/{seed}/`,
using the final coefficients after config and command-line overrides. For
example, `--tr_policy_coef 0.1 --tr_value_coef 0.05 --seed 23` uses
`logs/{domain}_{task}/trsppo_policy_0.1_value_0.05/23/`. Both TR coefficients
must be finite and nonnegative; the critic symmetry coefficient is separate
from PPO's ordinary `value_coef`.

For each real transition `(obs, action, reward, next_obs)`, `T(obs)` uses the
mapping in the table above. The actor's deterministic action is
`f(obs) = tanh(actor_mean(obs))`. The auxiliary loss is:

```text
tr_policy_loss = mean((f(T(next_obs)) - stop_gradient(f(obs))) ** 2)
actor_total_loss = actor_loss - entropy_coef * entropy + tr_policy_coef * tr_policy_loss
```

The mean includes valid transitions and all action components, using the action
dimension in the table above. The target
is recomputed by the current actor with gradients detached for each minibatch;
it is not the stored sampled action. Only the reversed-next-observation branch
receives the consistency gradient. PPO probabilities/returns/advantages are
computed only from the real rollout. No observation history or TR warmup is used.

For the critic ablation, use the same adjacent-state pairs as the actor:

```text
tr_value_loss = mean((V(T(next_obs)) - stop_gradient(V(obs))) ** 2)
critic_total_loss = value_coef * critic_loss + tr_value_coef * tr_value_loss
```

Here `critic_loss` is the ordinary MSE against GAE return targets. The reference
reuses the current critic predictions from that loss and receives no gradient
from the symmetry term. The additional gradient flows through `V(T(next_obs))`.
`--tr_value_coef` defaults to **0**; zero skips the transformation and additional
critic forward pass entirely, and logs zero for the skipped penalty. Positive
values encourage critic consistency across reversed transition endpoints as an
ablation. The actor and critic use the same valid-transition mask, selecting
eligible pairs before evaluating reversed observations.

`TRSPPORolloutBuffer` stores the actual next observation before an environment
reset and keeps it paired with the original observation during shuffling. Both
terminal and time-limit transitions are conservatively excluded from both TR
losses; they remain in PPO training, including timeout bootstrapping. A minibatch
with no eligible transitions contributes zero actor and critic TR loss.

`--tr_policy_coef` defaults to **0.1**. Set both TR coefficients to `0` for an
ablation that has vanilla PPO's parameter updates; the unweighted actor symmetry error is still logged
without gradients. All existing PPO settings, including the current
`rollout_steps=1000` default, apply. Do not pass `--time_rev`: that flag belongs
to SAC's synthetic replay augmentation.

TRSPPO saves the usual training/evaluation logs, TensorBoard events, videos and
configuration. Additional training scalars are `tr_policy_loss`,
`tr_policy_loss_weighted`, `tr_valid_pair_fraction`, `actor_total_loss`,
`tr_value_loss`, `tr_value_loss_weighted`, and `critic_total_loss`.
Update logs average minibatch diagnostics, weighted by minibatch size; each
minibatch's actor and critic TR losses are averaged over their valid pairs.
Checkpoints use
`model/trsppo_<step>.pt` and restore through `TRSPPOAgent.load(model_dir, step)`.
Construct the agent with the `domain_name`, `task_name`, network settings and
coefficients in `args.json`. Older checkpoints for existing domains can still be
loaded without `task_name`. Omitting `domain_name` retains the cheetah default.
The optional `buffer/rollout.npz` snapshot additionally includes `next_obs`.

Use the same observations, PPO hyperparameters, interaction budget and seeds
for PPO/TRSPPO comparisons. Acrobot swingup, cartpole swingup, reacher hard, both supported
pendulum tasks, walker stand and hopper stand have instantaneous rewards
invariant to velocity reversal with controls held fixed.
Walker walk/run, hopper hop and quadruped walk reward forward speed, so their
rewards are not invariant to velocity reversal. Finger spin rewards only spinner velocities at
or below -15 rad/s, so its reward is also asymmetric under time reversal.
The local acrobot model has zero joint damping
and disabled constraints; numerical integration still limits trajectory reversal
accuracy. The custom pendulum models also have zero joint damping and disabled
contacts, with numerical integration limiting exact reversal. Cartpole includes
small joint damping and a cart travel limit.
Walker, cheetah, hopper, finger and quadruped also include dissipative dynamics, so the
consistency relation is an experimental inductive bias rather than an exact
dynamics identity.

Reference comparison: the `xchen` branch of
[symm_rl_isaaclab](https://github.com/DLARlab/symm_rl_isaaclab/blob/3c34f5753969ab626280ea1c5eb8e8fc4b432761/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/time_reversal_ppo.py)
uses temporally shifted observation histories for both actor and critic
consistency. This implementation uses adjacent states for both losses, matching
the single-frame version of that temporal pairing without observation history.

Deferred paper work: develop a mathematical analysis of critic symmetry,
including invariant and sign-reversed rewards, physical time reversal versus
forward MDP symmetries, and the roles of policy, discounting, and episode
boundaries. Complete the critic ablation implementation before this analysis.

## Results
Our method demonstrates significantly improved performance over the baseline SAC:pixel. It matches the state-of-the-art performance of model-based algorithms, such as PlaNet (Hafner et al., 2018) and SLAC (Lee et al., 2019), as well
as a model-free algorithm D4PG (Barth-Maron et al., 2018), that also learns from raw images. Our
algorithm exhibits stable learning across ten random seeds and is extremely easy to implement.
![Results](results/graph.png)
