"""
Exploratory Data Analysis (EDA) for driving data.

Generates visualizations and statistics to validate data quality
before training.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_eda(data_dir: str, output_dir: str = "eda_output"):
    """
    Run full exploratory data analysis on collected driving data.

    Args:
        data_dir: Directory containing session CSV files.
        output_dir: Directory to save plots and reports.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    all_files = sorted(
        [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        ]
    )

    if not all_files:
        print("No CSV files found. Collect data first.")
        return

    frames = []
    for f in all_files:
        df = pd.read_csv(f)
        df["session"] = os.path.basename(f)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    obs_cols = [c for c in data.columns if c.startswith("obs_")]
    action_cols = [c for c in data.columns if c.startswith("action_")]

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Dataset overview
    _print_overview(data, all_files, obs_cols, action_cols)

    # 2. Missing values
    _check_missing(data, output_dir)

    # 3. Action distributions
    _plot_action_distributions(data, action_cols, output_dir)

    # 4. Observation distributions
    _plot_observation_distributions(data, obs_cols, output_dir)

    # 5. Correlation analysis
    _plot_correlations(data, obs_cols, action_cols, output_dir)

    # 6. Time series of actions
    _plot_action_timeseries(data, action_cols, output_dir)

    # 7. Per-session statistics
    _plot_session_stats(data, action_cols, output_dir)

    # 8. Observation heatmap
    _plot_observation_heatmap(data, obs_cols, output_dir)

    print(f"\nAll plots saved to: {output_dir}/")


def _print_overview(data, files, obs_cols, action_cols):
    """Print dataset overview statistics."""
    print(f"\nDataset Overview:")
    print(f"  Sessions: {len(files)}")
    print(f"  Total samples: {len(data)}")
    print(f"  Observation features: {len(obs_cols)}")
    print(f"  Action features: {len(action_cols)}")
    print(f"\nAction Statistics:")
    for col in action_cols:
        print(
            f"  {col}: mean={data[col].mean():.4f}, "
            f"std={data[col].std():.4f}, "
            f"min={data[col].min():.4f}, "
            f"max={data[col].max():.4f}"
        )
    print(f"\nObservation Statistics:")
    for col in obs_cols:
        print(
            f"  {col}: mean={data[col].mean():.4f}, "
            f"std={data[col].std():.4f}, "
            f"min={data[col].min():.4f}, "
            f"max={data[col].max():.4f}"
        )


def _check_missing(data, output_dir):
    """Check for missing values."""
    missing = data.isnull().sum()
    total_missing = missing.sum()
    print(f"\nMissing Values: {total_missing}")
    if total_missing > 0:
        print(missing[missing > 0])


def _plot_action_distributions(data, action_cols, output_dir):
    """Plot histograms of action distributions."""
    n_actions = len(action_cols)
    fig, axes = plt.subplots(1, n_actions, figsize=(6 * n_actions, 4))
    if n_actions == 1:
        axes = [axes]

    action_names = _get_action_names(n_actions)

    for i, col in enumerate(action_cols):
        axes[i].hist(data[col], bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(f"Distribution: {action_names[i]}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].axvline(x=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_distributions.png"), dpi=150)
    plt.close()
    print("  Saved: action_distributions.png")


def _plot_observation_distributions(data, obs_cols, output_dir):
    """Plot observation value distributions."""
    n_obs = len(obs_cols)
    n_cols_plot = min(5, n_obs)
    n_rows_plot = (n_obs + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(
        n_rows_plot, n_cols_plot,
        figsize=(4 * n_cols_plot, 3 * n_rows_plot),
    )
    if n_rows_plot == 1 and n_cols_plot == 1:
        axes = np.array([[axes]])
    elif n_rows_plot == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_plot == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(obs_cols):
        r, c = divmod(i, n_cols_plot)
        axes[r, c].hist(data[col], bins=30, edgecolor="black", alpha=0.7)
        axes[r, c].set_title(col, fontsize=8)

    # Hide unused axes
    for i in range(n_obs, n_rows_plot * n_cols_plot):
        r, c = divmod(i, n_cols_plot)
        axes[r, c].set_visible(False)

    plt.suptitle("Observation Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "observation_distributions.png"), dpi=150
    )
    plt.close()
    print("  Saved: observation_distributions.png")


def _plot_correlations(data, obs_cols, action_cols, output_dir):
    """Plot correlation matrix between observations and actions."""
    # Select a subset of observations if too many
    if len(obs_cols) > 20:
        step = len(obs_cols) // 20
        selected_obs = obs_cols[::step]
    else:
        selected_obs = obs_cols

    cols = list(selected_obs) + list(action_cols)
    corr = data[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=len(cols) <= 15,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close()
    print("  Saved: correlation_matrix.png")


def _plot_action_timeseries(data, action_cols, output_dir):
    """Plot time series of actions for a sample window."""
    window = min(1000, len(data))
    sample = data.head(window)

    action_names = _get_action_names(len(action_cols))
    n_actions = len(action_cols)

    fig, axes = plt.subplots(n_actions, 1, figsize=(14, 3 * n_actions), sharex=True)
    if n_actions == 1:
        axes = [axes]

    for i, col in enumerate(action_cols):
        axes[i].plot(sample.index, sample[col], linewidth=0.8)
        axes[i].set_ylabel(action_names[i])
        axes[i].set_ylim(-1.1, 1.1)
        axes[i].axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Sample Index")
    plt.suptitle(f"Action Time Series (first {window} samples)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_timeseries.png"), dpi=150)
    plt.close()
    print("  Saved: action_timeseries.png")


def _plot_session_stats(data, action_cols, output_dir):
    """Plot per-session sample counts."""
    if "session" not in data.columns:
        return

    session_counts = data.groupby("session").size()

    fig, ax = plt.subplots(figsize=(10, 4))
    session_counts.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Samples per Session")
    ax.set_xlabel("Session")
    ax.set_ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "session_stats.png"), dpi=150)
    plt.close()
    print("  Saved: session_stats.png")


def _plot_observation_heatmap(data, obs_cols, output_dir):
    """Plot heatmap of observation values over time (sampled)."""
    # Sample for visibility
    step = max(1, len(data) // 500)
    sampled = data[obs_cols].iloc[::step].values

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        sampled.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Sample (subsampled)")
    ax.set_ylabel("Ray Index")
    ax.set_title("Observation Values Over Time")
    plt.colorbar(im, ax=ax, label="Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "observation_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: observation_heatmap.png")


def _get_action_names(n_actions):
    """Return human-readable action names."""
    default_names = ["Steering", "Throttle"]
    if n_actions <= len(default_names):
        return default_names[:n_actions]
    return [f"Action {i}" for i in range(n_actions)]
