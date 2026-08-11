import pickle

import numpy as np

from src_continuous_control.plots.plot_simple_drift import make_plot


def _write_run(root, agent):
    returns = np.asarray([[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]], dtype=np.float32)
    with open(root / f"{agent}_ppo_seed_0_returns.pkl", "wb") as handle:
        pickle.dump(returns, handle)
    scalars = {
        "drift/multiplier": returns,
        "train/value_perm_l2": returns + 1.0,
        "train/value_trans_l2": returns * 0.1,
        "train/kl_prior": returns * 0.01,
        "train/consol/absorbed_frac": returns * 0.2,
    }
    with open(root / f"{agent}_ppo_seed_0_scalars.pkl", "wb") as handle:
        pickle.dump(scalars, handle)


def test_simple_drift_plot_writes_learning_figure(tmp_path):
    _write_run(tmp_path, "vanilla")
    _write_run(tmp_path, "pt_full")
    paths = make_plot(str(tmp_path), ["vanilla", "pt_full"], [0], str(tmp_path / "figures"))
    assert {path.rsplit(".", 1)[-1] for path in paths} == {"png", "pdf"}
    assert all((tmp_path / "figures" / path.rsplit("/", 1)[-1]).exists() for path in paths)
