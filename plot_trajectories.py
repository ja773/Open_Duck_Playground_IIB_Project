# plot_trajectories.py

import json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/0.5slip_clean_slope_eval_new_plots")

figures_dir = RESULTS_DIR / "figures"
figures_dir.mkdir(exist_ok=True)

for traj_file in RESULTS_DIR.glob("*_trajectory.json"):
    with open(traj_file) as f:
        traj = json.load(f)

    x = [p["x"] for p in traj]
    y = [p["y"] for p in traj]

    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.axhline(0.75, linestyle="--", linewidth=1)
    plt.axhline(-0.75, linestyle="--", linewidth=1)
    plt.xlabel("Forward position, x (m)")
    plt.ylabel("Lateral position, y (m)")
    plt.title(traj_file.stem.replace("_", " "))
    plt.axis("equal")
    plt.tight_layout()

    out = figures_dir / f"{traj_file.stem}.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved {out}")