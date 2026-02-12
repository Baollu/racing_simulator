"""
Racing Simulator AI - Main Entry Point

Simulator binary auto-detected at:
    RacingSimulatorLinux/BuildLinux/RacingSimulator.x86_64

Commands:
    collect  - Drive manually and collect training data
    eda      - Run exploratory data analysis on collected data
    train    - Train the AI model on collected data
    drive    - Let the AI drive autonomously
    config   - Validate agent configuration file

Usage:
    python main.py collect                          # auto-detects simulator
    python main.py collect --track circuit1         # name the track
    python main.py collect --input controller       # use gamepad
    python main.py eda                              # analyze collected data
    python main.py train                            # train model
    python main.py drive                            # AI drives
"""

import argparse
import os
import sys

# Auto-detect simulator binary relative to project root
SIM_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "RacingSimulatorLinux", "BuildLinux", "RacingSimulator.x86_64",
)

def find_simulator_path(user_path: str = None) -> str:
    """Resolve simulator binary path."""
    if user_path:
        path = user_path
    elif os.path.isfile(SIM_DEFAULT_PATH):
        path = SIM_DEFAULT_PATH
    else:
        print("Error: Simulator binary not found.")
        sys.exit(1)

    if not os.path.isfile(path):
        print(f"Error: Simulator binary not found at: {path}")
        sys.exit(1)

    # Ensure executable permission
    if not os.access(path, os.X_OK):
        print(f"Setting executable permission on {path}")
        os.chmod(path, os.stat(path).st_mode | 0o755)

    return os.path.abspath(path)


def cmd_collect(args):
    """Run manual driving with data collection."""
    from src.client import SimClient
    from src.data_collector import DataCollector
    from src.input_manager import create_input_manager

    import numpy as np

    # Validate config
    SimClient.load_agent_config(args.config)

    # Create input manager
    input_mgr = create_input_manager(
        mode=args.input,
        smoothing=args.smoothing,
    )

    # Resolve simulator path
    sim_path = find_simulator_path(args.sim_path)
    print(f"Simulator: {sim_path}")

    # Create client
    client = SimClient(
        config_path=args.config,
        sim_path=sim_path,
        base_port=args.port,
        time_scale=args.time_scale,
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
    )

    collector = DataCollector(data_dir=args.data_dir)

    try:
        client.connect()
        input_mgr.start()

        behavior_names = client.get_behavior_names()
        if not behavior_names:
            print("Error: No behaviors found.")
            return

        behavior_name = behavior_names[0]
        spec = client.get_behavior_spec(behavior_name)

        # Determine observation and action sizes
        obs_size = sum(
            np.prod(o.shape) for o in spec.observation_specs
        )
        action_size = spec.action_spec.continuous_size

        print(f"\nBehavior: {behavior_name}")
        print(f"Observation size: {obs_size}")
        print(f"Action size: {action_size}")

        collector.start_session(
            obs_size=obs_size,
            action_size=action_size,
            track_name=args.track,
        )

        print("\nDriving... Use keyboard/controller to steer.")
        print("Press Q to stop recording.\n")

        step_count = 0

        while not input_mgr.quit_requested:
            observations, agent_ids = client.get_observations(behavior_name)

            if observations is None:
                client.step()
                continue

            # Get human action
            action = input_mgr.get_action()

            for i, agent_id in enumerate(agent_ids):
                # Flatten all observations for this agent
                obs = np.concatenate(
                    [o[i].flatten() for o in observations]
                )

                # Record data
                collector.record(obs, action)

                # Send action to simulator
                action_2d = action.reshape(1, -1)
                client.set_actions(behavior_name, action_2d)

            client.step()
            step_count += 1

            if step_count % 100 == 0:
                print(
                    f"  Step {step_count} | "
                    f"Samples: {collector.sample_count} | "
                    f"Steering: {action[0]:+.2f} | "
                    f"Throttle: {action[1]:+.2f}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        input_mgr.stop()
        collector.end_session()
        client.close()


def cmd_eda(args):
    """Run exploratory data analysis."""
    from src.eda import run_eda

    run_eda(data_dir=args.data_dir, output_dir=args.output_dir)


def cmd_train(args):
    """Train the AI model."""
    from src.train import train_model

    train_model(
        data_dir=args.data_dir,
        config_path=args.training_config,
        model_dir=args.model_dir,
        device=args.device,
    )


def cmd_drive(args):
    """Run the AI driver."""
    sim_path = find_simulator_path(args.sim_path)

    if args.onnx:
        from src.drive import run_ai_driver_with_onnx

        run_ai_driver_with_onnx(
            config_path=args.config,
            onnx_path=args.model,
            model_dir=args.model_dir,
            sim_path=sim_path,
            base_port=args.port,
            time_scale=args.time_scale,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
            fullscreen=args.fullscreen,
        )
    else:
        from src.drive import run_ai_driver

        run_ai_driver(
            config_path=args.config,
            model_path=args.model,
            model_dir=args.model_dir,
            sim_path=sim_path,
            base_port=args.port,
            time_scale=args.time_scale,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
            fullscreen=args.fullscreen,
        )


def cmd_config(args):
    """Validate configuration file."""
    from src.client import SimClient

    try:
        config = SimClient.load_agent_config(args.config)
        print("Configuration is valid.")
        print(f"  Number of agents: {len(config['agents'])}")
        for i, agent in enumerate(config["agents"]):
            print(f"  Agent {i}: fov={agent['fov']}, nbRay={agent['nbRay']}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Racing Simulator AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- collect ---
    p_collect = subparsers.add_parser(
        "collect", help="Drive manually and collect training data"
    )
    p_collect.add_argument(
        "--config",
        default="config/agents_config.json",
        help="Path to agent config JSON (default: config/agents_config.json)",
    )
    p_collect.add_argument(
        "--sim-path",
        default=None,
        help="Path to RacingSimulator binary (default: auto-detect in RacingSimulatorLinux/)",
    )
    p_collect.add_argument(
        "--port", type=int, default=5004, help="Communication port (default: 5004)"
    )
    p_collect.add_argument(
        "--data-dir", default="data", help="Directory to save data (default: data/)"
    )
    p_collect.add_argument(
        "--track", default="default", help="Track name for file naming"
    )
    p_collect.add_argument(
        "--input",
        choices=["keyboard", "controller"],
        default="keyboard",
        help="Input method (default: keyboard)",
    )
    p_collect.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="Input smoothing factor 0-1 (default: 0.3 keyboard, 0.1 controller)",
    )
    p_collect.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Simulation time scale (default: 1.0)",
    )
    p_collect.add_argument(
        "--width", type=int, default=0,
        help="Window width in pixels (default: 80%% of screen)",
    )
    p_collect.add_argument(
        "--height", type=int, default=0,
        help="Window height in pixels (default: 80%% of screen)",
    )
    p_collect.add_argument(
        "--fullscreen", action="store_true",
        help="Launch simulator in fullscreen mode",
    )
    p_collect.set_defaults(func=cmd_collect)

    # --- eda ---
    p_eda = subparsers.add_parser(
        "eda", help="Run exploratory data analysis"
    )
    p_eda.add_argument(
        "--data-dir", default="data", help="Directory with CSV data files"
    )
    p_eda.add_argument(
        "--output-dir", default="eda_output", help="Directory for EDA plots"
    )
    p_eda.set_defaults(func=cmd_eda)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train the AI model")
    p_train.add_argument(
        "--data-dir", default="data", help="Directory with CSV data files"
    )
    p_train.add_argument(
        "--training-config",
        default="config/training_config.json",
        help="Training configuration JSON",
    )
    p_train.add_argument(
        "--model-dir", default="models", help="Directory to save models"
    )
    p_train.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device (default: auto)",
    )
    p_train.set_defaults(func=cmd_train)

    # --- drive ---
    p_drive = subparsers.add_parser("drive", help="Run AI autonomous driving")
    p_drive.add_argument(
        "--config",
        default="config/agents_config.json",
        help="Path to agent config JSON",
    )
    p_drive.add_argument(
        "--model",
        default="models/best_model.pth",
        help="Path to trained model file",
    )
    p_drive.add_argument(
        "--model-dir",
        default="models",
        help="Directory with normalization.json",
    )
    p_drive.add_argument(
        "--sim-path",
        default=None,
        help="Path to RacingSimulator binary (default: auto-detect)",
    )
    p_drive.add_argument(
        "--port", type=int, default=5004, help="Communication port"
    )
    p_drive.add_argument(
        "--time-scale", type=float, default=1.0, help="Simulation time scale"
    )
    p_drive.add_argument(
        "--max-steps", type=int, default=0, help="Max steps (0=unlimited)"
    )
    p_drive.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX runtime instead of PyTorch",
    )
    p_drive.add_argument(
        "--width", type=int, default=0,
        help="Window width in pixels (default: 80%% of screen)",
    )
    p_drive.add_argument(
        "--height", type=int, default=0,
        help="Window height in pixels (default: 80%% of screen)",
    )
    p_drive.add_argument(
        "--fullscreen", action="store_true",
        help="Launch simulator in fullscreen mode",
    )
    p_drive.set_defaults(func=cmd_drive)

    # --- config ---
    p_config = subparsers.add_parser("config", help="Validate configuration")
    p_config.add_argument(
        "--config",
        default="config/agents_config.json",
        help="Path to agent config JSON",
    )
    p_config.set_defaults(func=cmd_config)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
