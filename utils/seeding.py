"""Consistent seeding across random / numpy / torch / env.

Mirrors the baseline's seeding block (e.g. control/minatar_crl/DQN.py lines ~88-90) but also
seeds the environment's action space and reset, which matters for continuous control.
"""
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_env(env, seed: int):
    """Seed a gymnasium env's reset + action space. Returns the first observation."""
    obs, _ = env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass
    return obs
