"""Expanded HP sweep + multi-seed confirmation on the point-mass proxy.

Stage 1: grid search (single seed) over a conservative grid.
Stage 2: multi-seed confirmation (seeds 0..2) for the top_k configs.

Results are saved under `results/` as pickled objects.
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


def run_one(cfg, seed=0, total_updates=120, n_steps=32, num_envs=1):
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
    parser.add_argument("--grid-updates", type=int, default=120)
    parser.add_argument("--multi-updates", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # grid
    lr_perms = [3e-5, 1e-4, 3e-4]
    rhos = [0.25, 0.5, 0.75]
    kl_coefs = [0.0, 1e-3, 1e-2]
    ks = [4, 8, 16]

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
    for lr, rho, kl, k in itertools.product(lr_perms, rhos, kl_coefs, ks):
        c = dict(base_cfg)
        c["lr_perm"] = lr
        c["rho"] = rho
        c["kl_prior_coef"] = kl
        c["k"] = k
        combos.append(c)

    os.makedirs("results", exist_ok=True)
    grid_results = []
    start = time.time()
    print(f"[hp_expanded] running grid of {len(combos)} configs")
    for i, cfg in enumerate(combos):
        print(f"[hp_expanded] {i+1}/{len(combos)} lr_perm={cfg['lr_perm']} rho={cfg['rho']} kl={cfg['kl_prior_coef']} k={cfg['k']}")
        val = run_one(cfg, seed=0, total_updates=args.grid_updates, n_steps=32, num_envs=1)
        print(f"[hp_expanded] result: {val:.3f}")
        grid_results.append((cfg, float(val)))

    grid_results.sort(key=lambda x: x[1], reverse=True)
    with open("results/hp_sweep_proxy_expanded.pkl", "wb") as f:
        pickle.dump(grid_results, f)

    print("\n=== Grid results (best first) ===")
    for cfg, val in grid_results[: args.top_k * 2]:
        print(f"lr_perm={cfg['lr_perm']}, rho={cfg['rho']}, kl={cfg['kl_prior_coef']}, k={cfg['k']} -> {val:.3f}")

    # Multi-seed confirmation of top-k
    top_k = grid_results[: args.top_k]
    multi_results = []
    seeds = [0, 1, 2]
    print(f"\n[hp_expanded] multi-seed confirm top {len(top_k)} configs (seeds={seeds})")
    for cfg, _ in top_k:
        vals = []
        for s in seeds:
            print(f"[hp_expanded] top cfg lr_perm={cfg['lr_perm']} rho={cfg['rho']} seed={s}")
            v = run_one(cfg, seed=s, total_updates=args.multi_updates, n_steps=32, num_envs=1)
            print(f"[hp_expanded] seed {s} -> {v:.3f}")
            vals.append(float(v))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        multi_results.append((cfg, mean, std, vals))

    with open("results/hp_sweep_proxy_top_multiseed.pkl", "wb") as f:
        pickle.dump(multi_results, f)

    print("\n=== Multi-seed confirmation ===")
    for cfg, mean, std, vals in multi_results:
        print(f"lr_perm={cfg['lr_perm']}, rho={cfg['rho']}, kl={cfg['kl_prior_coef']}, k={cfg['k']} -> mean={mean:.3f} std={std:.3f} vals={vals}")

    print(f"\nFinished in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
