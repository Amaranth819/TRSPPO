"""Reset-safe histories used only when training with TimeReversalPPO."""

import gym
import torch


class TimeReversalHistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        from go2_gym_learn.ppo_cse.time_reversal import observation_layout

        super().__init__(env)
        self.layout = observation_layout(env.num_obs)
        self.obs_history_length = env.cfg.env.num_observation_history
        self.num_obs_history = self.obs_history_length * env.num_obs
        self.obs_history = torch.zeros(env.num_envs, self.num_obs_history, device=env.device)
        self._previous_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._obs = None

    def _append(self, obs, done):
        # The base environment resets internally and can expose the previous
        # episode's outgoing action on its first two observations. Sanitize only
        # our copies, leaving the original environment/standard PPO unchanged.
        obs = obs.clone()
        obs[self._previous_done, self.layout.second_previous_action] = 0.0
        obs[done, self.layout.previous_action] = 0.0
        obs[done, self.layout.second_previous_action] = 0.0
        history = self.obs_history.clone()
        history[done] = 0.0
        self.obs_history = torch.cat((history[:, self.env.num_obs:], obs), dim=-1)
        self._previous_done = done.clone()
        self._obs = obs

    def _observations(self, privileged):
        return {"obs": self._obs, "privileged_obs": privileged, "obs_history": self.obs_history}

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        self._append(obs, done.bool())
        return self._observations(info["privileged_obs"]), reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._append(obs, torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device))
        return self._observations(self.env.get_privileged_observations())

    def get_observations(self):
        # Reads must not append the same physical frame a second time.
        if self._obs is None:
            return self.reset()
        return self._observations(self.env.get_privileged_observations())

    def reset_idx(self, env_ids):
        result = self.env.reset_idx(env_ids)
        self.obs_history = self.obs_history.clone()
        self.obs_history[env_ids] = 0.0
        self._previous_done[env_ids] = True
        if self._obs is not None:
            self._obs = self._obs.clone()
            self._obs[env_ids] = 0.0
        return result
