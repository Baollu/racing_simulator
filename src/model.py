"""
Neural network model for supervised driving.

A small, efficient feedforward network that predicts steering and throttle
from raycast observations. Designed to be lightweight for deployment
on constrained hardware (Jetson Nano).
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn


class DrivingModel(nn.Module):
    """
    Feedforward neural network for predicting driving actions
    from raycast observations.

    Architecture:
        Input -> [Linear -> ReLU -> Dropout] x N -> Linear -> Tanh
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 2,
        hidden_sizes: list = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_size: Number of observation features.
            output_size: Number of action outputs (default 2: steering, throttle).
            hidden_sizes: List of hidden layer sizes (default [64, 32]).
            dropout: Dropout rate for regularization.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())  # Bound outputs to [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Action predictions of shape (batch_size, output_size).
        """
        return self.network(x)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Run inference on a single observation (no gradient).

        Args:
            observation: Numpy array of shape (input_size,) or (1, input_size).

        Returns:
            Action numpy array of shape (output_size,).
        """
        self.eval()
        with torch.no_grad():
            obs = observation.flatten()
            x = torch.FloatTensor(obs).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            action = self.network(x)
            return action.cpu().numpy().flatten()

    def save(self, path: str):
        """Save model weights and architecture config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_sizes": self.hidden_sizes,
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DrivingModel":
        """
        Load model from a checkpoint file.

        Args:
            path: Path to the saved model file.
            device: Device to load onto ("cpu" or "cuda").

        Returns:
            Loaded DrivingModel instance.
        """
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_size=state["input_size"],
            output_size=state["output_size"],
            hidden_sizes=state["hidden_sizes"],
        )
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model

    def export_onnx(self, path: str):
        """
        Export model to ONNX format for deployment on Jetson Nano.

        Args:
            path: Output path for the ONNX file.
        """
        self.eval()
        dummy_input = torch.randn(1, self.input_size)
        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes={
                "observations": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
            opset_version=11,
        )
        print(f"ONNX model exported to {path}")

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        print(f"DrivingModel:")
        print(f"  Input size:  {self.input_size}")
        print(f"  Output size: {self.output_size}")
        print(f"  Hidden:      {self.hidden_sizes}")
        print(f"  Parameters:  {self.count_parameters()}")
        print(f"  Architecture:")
        for name, module in self.network.named_children():
            print(f"    [{name}] {module}")
