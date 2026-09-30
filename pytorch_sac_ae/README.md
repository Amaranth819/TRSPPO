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
    --work_dir ./log \
    --seed 1
```
This will produce 'log' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
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

## PPO with state observations

Select `--agent ppo` to train separate actor and state-value MLPs, each with two
`hidden_dim` hidden layers and tanh activations. The Gaussian actor has learned,
state-independent log standard deviations and tanh-squashed actions in [-1, 1].
There is no encoder or decoder. SAC remains the default agent.

From `pytorch_sac_ae/` in the `tsda` environment:

```bash
python -u train.py --agent ppo \
    --domain_name cheetah --task_name run --action_repeat 4 \
    --num_train_steps 1000000 --seed 1 --work_dir ./log_cheetah_ppo \
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
| `--rollout_steps` | 2048 | Agent steps collected before updating |
| `--ppo_epochs` | 10 | Maximum passes over each rollout |
| `--gae_lambda` | 0.95 | GAE trace decay |
| `--ppo_clip` | 0.2 | Policy probability-ratio clipping range |
| `--entropy_coef` | 0.0 | Squashed-policy entropy bonus weight |
| `--value_coef` | 0.5 | Critic MSE weight |
| `--max_grad_norm` | 0.5 | Gradient norm limit for each network |
| `--target_kl` | 0.03 | Stop updates when minibatch approximate KL exceeds this; 0 disables |

PPO does not use SAC's `init_steps`, replay capacity, target-network settings,
update-frequency settings, or temperature settings. Pixel encoders/decoders,
prioritized replay, time reversal, and `gym_env` are rejected for this initial
PPO implementation. Training starts with policy samples, without random warmup.

`num_train_steps`, `rollout_steps` and logged steps count agent actions. With
`action_repeat=4`, each action sums up to four environment rewards. Evaluation
uses deterministic actions in a separate environment seeded with `seed + 10000`,
at step 0, every `eval_freq` steps, and at the final training step.

Outputs under `work_dir`:

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
Use a distinct `work_dir` for each run: the existing logger replaces previous
train/evaluation logs and TensorBoard events in that directory.

Checkpoints can be loaded with `PPOAgent.load(model_dir, step)` into an agent
constructed with the same network settings. This restores networks and
optimizers; it does not resume environment state or an in-progress rollout.

To run the focused PPO tests from the repository root:

```bash
python -m unittest discover -s pytorch_sac_ae -p test_ppo.py -v
```

## Results
Our method demonstrates significantly improved performance over the baseline SAC:pixel. It matches the state-of-the-art performance of model-based algorithms, such as PlaNet (Hafner et al., 2018) and SLAC (Lee et al., 2019), as well
as a model-free algorithm D4PG (Barth-Maron et al., 2018), that also learns from raw images. Our
algorithm exhibits stable learning across ten random seeds and is extremely easy to implement.
![Results](results/graph.png)
