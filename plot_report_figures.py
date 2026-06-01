from pathlib import Path
import json
import re

import matplotlib.pyplot as plt


BASELINE_DIR = Path("results/baseline_clean_slope_eval_new_plots")
SLIP_DIR = Path("results/0.5slip_clean_slope_eval_new_plots")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


CORRIDOR = 0.75


def load_summary(results_dir):
    with open(results_dir / "summary.json") as f:
        return json.load(f)


def load_traj(results_dir, experiment, trial=0):
    path = results_dir / f"{experiment}_trial_{trial}_trajectory.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")
    with open(path) as f:
        return json.load(f)


def plot_single_traj(ax, traj, title, label=None):
    x = [p["x"] for p in traj]
    y = [p["y"] for p in traj]

    # Shift to start at origin for easier comparison.
    x0, y0 = x[0], y[0]
    x = [v - x0 for v in x]
    y = [v - y0 for v in y]

    ax.plot(x, y, linewidth=1.6, label=label)
    ax.axhline(CORRIDOR, linestyle="--", linewidth=1)
    ax.axhline(-CORRIDOR, linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Forward displacement, x (m)")
    ax.set_ylabel("Lateral displacement, y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)


def save_baseline_trajectory_analysis():
    experiments = [
        ("flat", "Flat terrain"),
        ("up_1deg", r"Uphill $1^\circ$"),
        ("down_3deg", r"Downhill $3^\circ$"),
        ("down_7deg", r"Downhill $7^\circ$"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(8, 7))
    axs = axs.flatten()

    for ax, (exp, title) in zip(axs, experiments):
        traj = load_traj(BASELINE_DIR, exp, trial=0)
        plot_single_traj(ax, traj, title)

    fig.suptitle("Baseline policy trajectories", y=0.995)
    fig.tight_layout()
    out = FIG_DIR / "baseline_trajectory_analysis.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def save_flat_baseline_vs_slip():
    fig, ax = plt.subplots(figsize=(5.8, 4.5))

    plot_single_traj(
        ax,
        load_traj(BASELINE_DIR, "flat", trial=0),
        "Flat terrain",
        label="Baseline",
    )

    plot_single_traj(
        ax,
        load_traj(SLIP_DIR, "flat", trial=0),
        "Flat terrain",
        label="Slip 0.5",
    )

    ax.legend()
    out = FIG_DIR / "flat_baseline_vs_slip.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def save_up1_baseline_vs_slip():
    fig, ax = plt.subplots(figsize=(5.8, 4.5))

    plot_single_traj(
        ax,
        load_traj(BASELINE_DIR, "up_1deg", trial=0),
        r"Uphill $1^\circ$",
        label="Baseline",
    )

    plot_single_traj(
        ax,
        load_traj(SLIP_DIR, "up_1deg", trial=0),
        r"Uphill $1^\circ$",
        label="Slip 0.5",
    )

    ax.legend()
    out = FIG_DIR / "up1_baseline_vs_slip.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def save_downhill5_baseline_vs_slip():
    fig, ax = plt.subplots(figsize=(5.8, 4.5))

    plot_single_traj(
        ax,
        load_traj(BASELINE_DIR, "down_5deg", trial=0),
        r"Downhill $5^\circ$",
        label="Baseline",
    )

    plot_single_traj(
        ax,
        load_traj(SLIP_DIR, "down_5deg", trial=0),
        r"Downhill $5^\circ$",
        label="Slip 0.5",
    )

    ax.legend()
    out = FIG_DIR / "downhill5_baseline_vs_slip.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def signed_angle(row):
    if row["direction"] == "up":
        return row["angle_deg"]
    if row["direction"] == "down":
        return -row["angle_deg"]
    return 0


def save_success_vs_slope():
    baseline = load_summary(BASELINE_DIR)
    slip = load_summary(SLIP_DIR)

    def split(summary):
        pts = []
        for r in summary:
            pts.append((signed_angle(r), r["success_rate"]))
        pts = sorted(pts, key=lambda p: p[0])
        return [p[0] for p in pts], [p[1] for p in pts]

    bx, by = split(baseline)
    sx, sy = split(slip)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(bx, by, marker="o", label="Baseline")
    ax.plot(sx, sy, marker="s", label="Slip 0.5")

    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Slope angle (deg), negative = downhill")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linewidth=0.3)
    ax.legend()

    out = FIG_DIR / "success_vs_slope.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    save_baseline_trajectory_analysis()
    save_flat_baseline_vs_slip()
    save_up1_baseline_vs_slip()
    save_downhill5_baseline_vs_slip()
    save_success_vs_slope()


if __name__ == "__main__":
    main()