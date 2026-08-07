"""Focused HP sweep for the point-mass proxy.

Runs lr_perm ∈ {1e-4,2e-4,3e-4} × k ∈ {8,16} with rho=0.5, multi-seed (default 5).
Saves results to `results/hp_focused_proxy_multiseed.pkl`.
"""
import argparse
import itertools
import os
import pickle
import time

import numpy as np

from envs.mock_continual import make_directional_point_vector_env
from agents.ppo_pt_full import PPOPTFull
from utils.seeding import seed_everything


def run_one(cfg, seed=0, total_updates=300, n_steps=32, num_envs=1):
    seed_everything(seed)
    env = make_directional_point_vector_env(direction=1.0, num_envs=num_envs,
                                            max_episode_steps=150, normalize_obs=False,
                                            normalize_reward=False)
    obs, _ = env.reset(seed=seed)
    done = np.zeros(num_envs, dtype=np.float32)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    cfg_local = dict(cfg)
    cfg_local["n_steps"] = n_steps
    cfg_local["num_envs"] = num_envs
    cfg_local.setdefault("lr_actor", 1e-3)
    agent = PPOPTFull(obs_dim, act_dim, cfg_local, device="cpu")

    avg_return = 0.0
    for update_idx in range(total_updates):
        obs, done, episode_returns = agent.collect_rollout(env, obs, done)
        _ = agent.update(obs, done, update_idx)
        for r in episode_returns:
            avg_return = 0.99 * avg_return + 0.01 * r
    return avg_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-updates", type=int, default=300)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--lr-perms", type=str, default="1e-4,2e-4,3e-4")
    parser.add_argument("--ks", type=str, default="8,16")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    lr_perms = [float(x) for x in args.lr_perms.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    rho = args.rho

    base_cfg = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "epochs": 3,
        "minibatch_size": 32,
        "ent_coef": 0.0,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "consolidation_epochs": 1,
        "critic_hidden_sizes": [64, 64],
        "hidden_sizes": [64, 64],
        "actor_trans_hidden_sizes": [32, 32],
        "critic_trans_hidden_sizes": [32, 32],
        "lr_trans": 1e-3,
        "perm_optimizer": "adam",
        "rm_power": 0.5,
        "consolidation_buffer_size": 1024,
    }

    combos = []
    for lr, k in itertools.product(lr_perms, ks):
        c = dict(base_cfg)
        c["lr_perm"] = lr
        c["rho"] = rho
        c["kl_prior_coef"] = 0.0
        c["k"] = k
        combos.append(c)

    os.makedirs("results", exist_ok=True)
    results = []
    start = time.time()
    seeds = list(range(args.num_seeds))
    print(f"[hp_focused] running {len(combos)} configs; seeds={seeds}")
    for i, cfg in enumerate(combos):
        print(f"[hp_focused] {i+1}/{len(combos)} lr_perm={cfg['lr_perm']} rho={cfg['rho']} k={cfg['k']}")
        vals = []
        for s in seeds:
            v = run_one(cfg, seed=s, total_updates=args.total_updates, n_steps=args.n_steps, num_envs=args.num_envs)
            print(f"[hp_focused]   seed {s} -> {v:.3f}")
            vals.append(float(v))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        results.append((cfg, mean, std, vals))

    with open("results/hp_focused_proxy_multiseed.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\n=== Focused multi-seed results ===")
    for cfg, mean, std, vals in results:
        print(f"lr_perm={cfg['lr_perm']}, rho={cfg['rho']}, kl={cfg['kl_prior_coef']}, k={cfg['k']} -> mean={mean:.3f} std={std:.3f} vals={vals}")
    print(f"\nFinished in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
