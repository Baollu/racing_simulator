"""
Data collector for recording driving sessions.

Records observation-action pairs during manual driving sessions
and saves them as CSV files for training. Supports session management,
data validation, and basic statistics.
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np


class DataCollector:
    """Records driving data (observations + actions) to CSV files."""

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Directory where session files are saved.
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self._session_file = None
        self._writer = None
        self._file_handle = None
        self._sample_count = 0
        self._session_name = None
        self._obs_size = None
        self._action_size = None
        self._header = None

    def start_session(
        self,
        obs_size: int,
        action_size: int,
        track_name: str = "unknown",
    ):
        """
        Start a new recording session.

        Args:
            obs_size: Size of the flattened observation vector.
            action_size: Size of the action vector.
            track_name: Name of the track for file naming.
        """
        self._obs_size = obs_size
        self._action_size = action_size
        self._sample_count = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_name = f"session_{track_name}_{timestamp}"
        filepath = os.path.join(self.data_dir, f"{self._session_name}.csv")

        # Build header
        obs_cols = [f"obs_{i}" for i in range(obs_size)]
        action_cols = [f"action_{i}" for i in range(action_size)]
        self._header = ["timestamp"] + obs_cols + action_cols

        self._file_handle = open(filepath, "w", newline="")
        self._writer = csv.writer(self._file_handle)
        self._writer.writerow(self._header)

        print(f"Recording session started: {filepath}")
        print(f"  Observation size: {obs_size}")
        print(f"  Action size: {action_size}")

    def record(self, observation: np.ndarray, action: np.ndarray):
        """
        Record a single observation-action pair.

        Args:
            observation: Flattened observation vector.
            action: Action vector.
        """
        if self._writer is None:
            raise RuntimeError("No active session. Call start_session() first.")

        obs_flat = observation.flatten()
        action_flat = action.flatten()

        if len(obs_flat) != self._obs_size:
            raise ValueError(
                f"Observation size mismatch: expected {self._obs_size}, "
                f"got {len(obs_flat)}"
            )
        if len(action_flat) != self._action_size:
            raise ValueError(
                f"Action size mismatch: expected {self._action_size}, "
                f"got {len(action_flat)}"
            )

        row = [time.time()] + obs_flat.tolist() + action_flat.tolist()
        self._writer.writerow(row)
        self._sample_count += 1

        # Flush periodically to avoid data loss
        if self._sample_count % 100 == 0:
            self._file_handle.flush()

    def end_session(self):
        """End the current recording session."""
        if self._file_handle is not None:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None
            self._writer = None

            print(f"Session ended. Recorded {self._sample_count} samples.")

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def session_name(self) -> Optional[str]:
        return self._session_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_session()
        return False


def load_dataset(data_dir: str, obs_size: Optional[int] = None):
    """
    Load all session CSV files from a directory into numpy arrays.

    Args:
        data_dir: Directory containing session CSV files.
        obs_size: Expected observation size. If None, inferred from
                  column names.

    Returns:
        Tuple of (observations, actions) as numpy arrays.
        observations shape: (N, obs_size)
        actions shape: (N, action_size)
    """
    import pandas as pd

    all_files = sorted(
        [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        ]
    )

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for filepath in all_files:
        df = pd.read_csv(filepath)
        if len(df) > 0:
            frames.append(df)
            print(f"  Loaded {filepath}: {len(df)} samples")

    if not frames:
        raise ValueError("All CSV files are empty.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Total samples loaded: {len(combined)}")

    # Separate observation and action columns
    obs_cols = [c for c in combined.columns if c.startswith("obs_")]
    action_cols = [c for c in combined.columns if c.startswith("action_")]

    if not obs_cols or not action_cols:
        raise ValueError(
            "CSV must have columns named 'obs_*' and 'action_*'."
        )

    observations = combined[obs_cols].values.astype(np.float32)
    actions = combined[action_cols].values.astype(np.float32)

    # Remove rows with NaN
    valid_mask = ~(np.isnan(observations).any(axis=1) | np.isnan(actions).any(axis=1))
    if not valid_mask.all():
        n_removed = (~valid_mask).sum()
        print(f"Removed {n_removed} rows with NaN values.")
        observations = observations[valid_mask]
        actions = actions[valid_mask]

    return observations, actions


def get_dataset_stats(data_dir: str) -> dict:
    """
    Get statistics about the collected dataset.

    Returns:
        Dictionary with dataset statistics.
    """
    import pandas as pd

    all_files = sorted(
        [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        ]
    )

    stats = {
        "num_files": len(all_files),
        "total_samples": 0,
        "files": [],
    }

    for filepath in all_files:
        df = pd.read_csv(filepath)
        file_info = {
            "name": os.path.basename(filepath),
            "samples": len(df),
        }
        stats["total_samples"] += len(df)
        stats["files"].append(file_info)

    return stats
