"""Figure 6.1: HalfCheetah under piecewise physics change, sigma learned, ten seeds per arm.

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_fig61_learned

Copies the per-seed return curves from results/clean/{vanilla,pt,ewc_fixed} into one flat directory
(the layout make_phase2_figures expects), draws the figure and saves it as
figures_phase2/a_return_vs_timestep_learned.{png,pdf}. Do not use results/hard (sigma fixed at 1.0).
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "results", "clean")
OUT = os.path.join(HERE, "figures_phase2")


def main():
    with tempfile.TemporaryDirectory() as flat, tempfile.TemporaryDirectory() as out:
        for sub in ("vanilla", "pt", "ewc_fixed"):
            d = os.path.join(CLEAN, sub)
            for f in os.listdir(d):
                if f.endswith("_returns.pkl"):
                    shutil.copy(os.path.join(d, f), flat)
        subprocess.run([sys.executable, "-m", "src_continuous_control.plots.make_phase2_figures",
                        "--seeds", *map(str, range(10)), "--results-dir", flat,
                        "--frozen-dir", os.path.join(CLEAN, "pt_frozen"), "--out-dir", out],
                       check=True)
        for ext in ("png", "pdf"):
            shutil.copy(os.path.join(out, f"a_return_vs_timestep.{ext}"),
                        os.path.join(OUT, f"a_return_vs_timestep_learned.{ext}"))
    print("wrote figures_phase2/a_return_vs_timestep_learned.png / .pdf")


if __name__ == "__main__":
    main()
