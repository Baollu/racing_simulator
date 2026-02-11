"""
AI driving module for autonomous control.

Loads a trained model and uses it to control the car in the simulator.
Supports both real-time inference and evaluation mode.
"""

import json
import os
import time

import numpy as np

from .client import SimClient
from .model import DrivingModel


def run_ai_driver(
    config_path: str,
    model_path: str,
    model_dir: str = "models",
    sim_path: str = None,
    base_port: int = 5004,
    time_scale: float = 1.0,
    max_steps: int = 0,
    width: int = 0,
    height: int = 0,
    fullscreen: bool = False,
):
    """
    Run the AI driver in the simulator.

    Args:
        config_path: Path to agent configuration JSON.
        model_path: Path to the trained model .pth file.
        model_dir: Directory containing normalization.json.
        sim_path: Path to the simulator binary (None for running instance).
        base_port: Communication port.
        time_scale: Simulation speed.
        max_steps: Maximum steps to run (0 = unlimited).
        width: Window width (0 = auto).
        height: Window height (0 = auto).
        fullscreen: Fullscreen mode.
    """
    # Load model
    device = "cpu"  # CPU for real-time inference (lower latency)
    model = DrivingModel.load(model_path, device=device)
    model.summary()

    # Load normalization parameters
    norm_path = os.path.join(model_dir, "normalization.json")
    obs_mean = None
    obs_std = None
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            norm = json.load(f)
        obs_mean = np.array(norm["mean"], dtype=np.float32)
        obs_std = np.array(norm["std"], dtype=np.float32)
        print("Normalization parameters loaded.")
    else:
        print("Warning: normalization.json not found. Using raw observations.")

    # Connect to simulator
    client = SimClient(
        config_path=config_path,
        sim_path=sim_path,
        base_port=base_port,
        time_scale=time_scale,
        width=width,
        height=height,
        fullscreen=fullscreen,
    )

    try:
        client.connect()
        behavior_names = client.get_behavior_names()

        if not behavior_names:
            print("Error: No behaviors found in simulation.")
            return

        behavior_name = behavior_names[0]
        print(f"\nAI driving on behavior: {behavior_name}")
        print("Press Ctrl+C to stop.\n")

        step_count = 0
        inference_times = []

        while True:
            observations, agent_ids = client.get_observations(behavior_name)

            if observations is None:
                client.step()
                continue

            # Flatten observations for each agent
            for i, agent_id in enumerate(agent_ids):
                obs = np.concatenate(
                    [o[i].flatten() for o in observations]
                )

                # Normalize
                if obs_mean is not None and obs_std is not None:
                    obs = (obs - obs_mean) / obs_std

                # Inference
                t0 = time.time()
                action = model.predict(obs)
                inference_time = (time.time() - t0) * 1000
                inference_times.append(inference_time)

                # Send action
                client.set_action_for_agent(
                    behavior_name, agent_id, action
                )

            client.step()
            step_count += 1

            # Periodic status
            if step_count % 200 == 0:
                avg_ms = np.mean(inference_times[-200:])
                print(
                    f"Step {step_count} | "
                    f"Avg inference: {avg_ms:.2f}ms"
                )

            if 0 < max_steps <= step_count:
                print(f"\nReached max steps ({max_steps}).")
                break

    except KeyboardInterrupt:
        print("\nStopping AI driver.")
    finally:
        if inference_times:
            print(f"\nInference statistics:")
            print(f"  Mean: {np.mean(inference_times):.2f}ms")
            print(f"  Median: {np.median(inference_times):.2f}ms")
            print(f"  Max: {np.max(inference_times):.2f}ms")
            print(f"  Total steps: {step_count}")
        client.close()


def run_ai_driver_with_onnx(
    config_path: str,
    onnx_path: str,
    model_dir: str = "models",
    sim_path: str = None,
    base_port: int = 5004,
    time_scale: float = 1.0,
    max_steps: int = 0,
    width: int = 0,
    height: int = 0,
    fullscreen: bool = False,
):
    """
    Run the AI driver using an ONNX model (for Jetson Nano deployment testing).

    Args:
        config_path: Path to agent configuration JSON.
        onnx_path: Path to the ONNX model file.
        model_dir: Directory containing normalization.json.
        sim_path: Path to the simulator binary.
        base_port: Communication port.
        time_scale: Simulation speed.
        max_steps: Maximum steps to run (0 = unlimited).
        width: Window width (0 = auto).
        height: Window height (0 = auto).
        fullscreen: Fullscreen mode.
    """
    import onnxruntime as ort

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    print(f"ONNX model loaded: {onnx_path}")

    # Load normalization parameters
    norm_path = os.path.join(model_dir, "normalization.json")
    obs_mean = None
    obs_std = None
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            norm = json.load(f)
        obs_mean = np.array(norm["mean"], dtype=np.float32)
        obs_std = np.array(norm["std"], dtype=np.float32)
        print("Normalization parameters loaded.")

    # Connect to simulator
    client = SimClient(
        config_path=config_path,
        sim_path=sim_path,
        base_port=base_port,
        time_scale=time_scale,
        width=width,
        height=height,
        fullscreen=fullscreen,
    )

    try:
        client.connect()
        behavior_names = client.get_behavior_names()

        if not behavior_names:
            print("Error: No behaviors found in simulation.")
            return

        behavior_name = behavior_names[0]
        print(f"\nAI driving (ONNX) on behavior: {behavior_name}")
        print("Press Ctrl+C to stop.\n")

        step_count = 0
        inference_times = []

        while True:
            observations, agent_ids = client.get_observations(behavior_name)

            if observations is None:
                client.step()
                continue

            for i, agent_id in enumerate(agent_ids):
                obs = np.concatenate(
                    [o[i].flatten() for o in observations]
                )

                if obs_mean is not None and obs_std is not None:
                    obs = (obs - obs_mean) / obs_std

                obs_input = obs.reshape(1, -1).astype(np.float32)

                t0 = time.time()
                result = session.run(None, {input_name: obs_input})
                action = result[0].flatten()
                inference_time = (time.time() - t0) * 1000
                inference_times.append(inference_time)

                client.set_action_for_agent(
                    behavior_name, agent_id, action
                )

            client.step()
            step_count += 1

            if step_count % 200 == 0:
                avg_ms = np.mean(inference_times[-200:])
                print(
                    f"Step {step_count} | "
                    f"Avg inference: {avg_ms:.2f}ms"
                )

            if 0 < max_steps <= step_count:
                break

    except KeyboardInterrupt:
        print("\nStopping AI driver.")
    finally:
        if inference_times:
            print(f"\nONNX Inference statistics:")
            print(f"  Mean: {np.mean(inference_times):.2f}ms")
            print(f"  Median: {np.median(inference_times):.2f}ms")
            print(f"  Max: {np.max(inference_times):.2f}ms")
            print(f"  Total steps: {step_count}")
        client.close()
