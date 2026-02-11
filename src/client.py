"""
Unity simulation client using mlagents_envs.

Wraps UnityEnvironment to provide a clean interface for connecting
to the Racing Simulator, reading observations, and sending actions.
"""

import json
import os
from typing import Optional

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)


class SimClient:
    """Client for connecting to the Racing Simulator via ML-Agents."""

    def __init__(
        self,
        config_path: str,
        sim_path: Optional[str] = None,
        base_port: int = 5004,
        worker_id: int = 0,
        no_graphics: bool = False,
        time_scale: float = 1.0,
    ):
        """
        Initialize the simulation client.

        Args:
            config_path: Path to the agents configuration JSON file.
            sim_path: Path to the RacingSimulator binary. If None, connects
                      to an already-running simulator instance.
            base_port: Base port for communication (default 5004).
            worker_id: Worker ID offset from base_port.
            no_graphics: If True, run simulator without graphics.
            time_scale: Simulation time scale (1.0 = real-time).
        """
        self.config_path = os.path.abspath(config_path)
        self.sim_path = sim_path
        self.base_port = base_port
        self.worker_id = worker_id
        self.no_graphics = no_graphics
        self.time_scale = time_scale

        self._env = None
        self._engine_channel = None
        self._behavior_names = []
        self._behavior_specs = {}

    def connect(self):
        """Connect to the Unity simulation."""
        self._engine_channel = EngineConfigurationChannel()

        additional_args = ["--config-path", self.config_path]

        self._env = UnityEnvironment(
            file_name=self.sim_path,
            base_port=self.base_port,
            worker_id=self.worker_id,
            no_graphics=self.no_graphics,
            side_channels=[self._engine_channel],
            additional_args=additional_args,
        )

        self._engine_channel.set_configuration_parameters(
            time_scale=self.time_scale,
        )

        self._env.reset()

        self._behavior_names = list(self._env.behavior_specs.keys())
        for name in self._behavior_names:
            self._behavior_specs[name] = self._env.behavior_specs[name]

        print(f"Connected to simulator. Behaviors: {self._behavior_names}")
        for name, spec in self._behavior_specs.items():
            print(f"  {name}:")
            print(f"    Observation shapes: {[s.shape for s in spec.observation_specs]}")
            print(f"    Action spec: {spec.action_spec}")

    def get_behavior_names(self) -> list:
        """Return list of behavior names (one per agent)."""
        return self._behavior_names

    def get_behavior_spec(self, behavior_name: str):
        """Return the BehaviorSpec for a given behavior."""
        return self._behavior_specs[behavior_name]

    def get_steps(self, behavior_name: str):
        """
        Get current decision and terminal steps for a behavior.

        Returns:
            Tuple of (DecisionSteps, TerminalSteps).
        """
        return self._env.get_steps(behavior_name)

    def get_observations(self, behavior_name: str):
        """
        Get observations for all agents of a behavior.

        Returns:
            List of numpy arrays (one per observation type),
            or None if no agents need decisions.
        """
        decision_steps, _ = self.get_steps(behavior_name)
        if len(decision_steps) == 0:
            return None, None
        return decision_steps.obs, decision_steps.agent_id

    def set_actions(self, behavior_name: str, actions: np.ndarray):
        """
        Send actions for all agents of a behavior.

        Args:
            actions: ActionTuple or numpy array of actions.
        """
        from mlagents_envs.base_env import ActionTuple

        if isinstance(actions, np.ndarray):
            action_tuple = ActionTuple(continuous=actions.astype(np.float32))
        else:
            action_tuple = actions

        self._env.set_actions(behavior_name, action_tuple)

    def set_action_for_agent(
        self, behavior_name: str, agent_id: int, action: np.ndarray
    ):
        """Send action for a single agent."""
        from mlagents_envs.base_env import ActionTuple

        action_2d = action.reshape(1, -1).astype(np.float32)
        action_tuple = ActionTuple(continuous=action_2d)
        self._env.set_action_for_agent(behavior_name, agent_id, action_tuple)

    def step(self):
        """Advance the simulation by one step."""
        self._env.step()

    def reset(self):
        """Reset the simulation."""
        self._env.reset()

    def close(self):
        """Close the connection to the simulation."""
        if self._env is not None:
            self._env.close()
            self._env = None
            print("Disconnected from simulator.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @staticmethod
    def load_agent_config(config_path: str) -> dict:
        """Load and validate agent configuration from JSON."""
        with open(config_path, "r") as f:
            config = json.load(f)

        if "agents" not in config:
            raise ValueError("Config must contain 'agents' key.")

        for i, agent in enumerate(config["agents"]):
            if "fov" not in agent or "nbRay" not in agent:
                raise ValueError(
                    f"Agent {i} must have 'fov' and 'nbRay' fields."
                )
            if not (1 <= agent["fov"] <= 180):
                raise ValueError(
                    f"Agent {i} fov must be between 1 and 180, got {agent['fov']}."
                )
            if not (1 <= agent["nbRay"] <= 50):
                raise ValueError(
                    f"Agent {i} nbRay must be between 1 and 50, got {agent['nbRay']}."
                )

        return config
