"""Explicit algorithm selection; RMA remains an alias for the original PPO."""


def normalize_algorithm(name):
    aliases = {"ppo": "ppo", "rma": "ppo", "trsppo": "trsppo"}
    try:
        return aliases[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError("Unknown algorithm %r; choose ppo or trsppo" % name)


def make_algorithm(name, actor_critic, device="cpu", time_reversal_config=None):
    if normalize_algorithm(name) == "ppo":
        from .ppo import PPO
        return PPO(actor_critic, device=device)
    from .time_reversal_ppo import TimeReversalPPO
    return TimeReversalPPO(actor_critic, device=device, config=time_reversal_config)
