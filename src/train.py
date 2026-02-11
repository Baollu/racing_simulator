"""
Training loop for the supervised driving model.

Handles data loading, train/validation split, training loop with
early stopping, evaluation metrics, and model checkpointing.
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .data_collector import load_dataset
from .model import DrivingModel


def train_model(
    data_dir: str,
    config_path: str = "config/training_config.json",
    model_dir: str = "models",
    device: str = "auto",
):
    """
    Train the driving model on collected data.

    Args:
        data_dir: Directory containing session CSV files.
        config_path: Path to training configuration JSON.
        model_dir: Directory to save model checkpoints.
        device: Device to train on ("cpu", "cuda", or "auto").
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 100)
    hidden_sizes = config.get("hidden_sizes", [64, 32])
    dropout = config.get("dropout", 0.1)
    val_split = config.get("validation_split", 0.2)
    patience = config.get("early_stopping_patience", 10)
    sched_step = config.get("scheduler_step_size", 30)
    sched_gamma = config.get("scheduler_gamma", 0.5)

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load data
    print("\nLoading dataset...")
    observations, actions = load_dataset(data_dir)
    input_size = observations.shape[1]
    output_size = actions.shape[1]

    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")

    # Normalize observations
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0)
    obs_std[obs_std == 0] = 1.0  # Avoid division by zero
    observations = (observations - obs_mean) / obs_std

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        observations, actions, test_size=val_split, random_state=42
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = DrivingModel(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )
    model.to(device)
    model.summary()

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=sched_step, gamma=sched_gamma
    )

    # Training loop
    os.makedirs(model_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"\nTraining for up to {epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Train phase
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        epoch_train_loss /= n_batches
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        epoch_val_loss /= n_val_batches
        val_losses.append(epoch_val_loss)

        scheduler.step()

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {epoch_val_loss:.6f} | "
                f"LR: {current_lr:.6f}"
            )

        # Early stopping & checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            model.save(os.path.join(model_dir, "best_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print("-" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model
    model.save(os.path.join(model_dir, "final_model.pth"))

    # Save normalization parameters
    norm_params = {"mean": obs_mean.tolist(), "std": obs_std.tolist()}
    norm_path = os.path.join(model_dir, "normalization.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f)
    print(f"Normalization params saved to {norm_path}")

    # Plot training curves
    _plot_training_curves(train_losses, val_losses, model_dir)

    # Final evaluation on validation set
    print("\n" + "=" * 60)
    print("EVALUATION ON VALIDATION SET")
    print("=" * 60)
    best_model = DrivingModel.load(
        os.path.join(model_dir, "best_model.pth"), device=device
    )
    evaluate_model(best_model, X_val, y_val, device=device)

    # Export ONNX
    onnx_path = os.path.join(model_dir, "model.onnx")
    best_model.to("cpu")
    best_model.export_onnx(onnx_path)

    return best_model


def evaluate_model(
    model: DrivingModel,
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
):
    """
    Evaluate model with comprehensive metrics.

    Args:
        model: Trained DrivingModel.
        X: Observation features (already normalized).
        y: True actions.
        device: Device for inference.
    """
    model.eval()
    model.to(device)

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    y_true = y
    y_pred = predictions

    # Overall metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nOverall Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")

    # Per-action metrics
    action_names = ["Steering", "Throttle"]
    for i in range(y_true.shape[1]):
        name = action_names[i] if i < len(action_names) else f"Action {i}"
        mse_i = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        print(f"\n  {name}:")
        print(f"    MSE:  {mse_i:.6f}")
        print(f"    RMSE: {np.sqrt(mse_i):.6f}")
        print(f"    MAE:  {mae_i:.6f}")
        print(f"    R²:   {r2_i:.6f}")


def _plot_training_curves(train_losses, val_losses, output_dir):
    """Plot and save training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation Loss", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Best: epoch {best_epoch}\nloss={best_loss:.6f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + len(train_losses) * 0.05, best_loss * 1.5),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {output_dir}/training_curves.png")
