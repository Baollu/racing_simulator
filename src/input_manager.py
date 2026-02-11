"""
Input management system for controlling the car.

Supports two backends:
- keyboard: Uses pynput for global keyboard capture (no window needed).
- controller: Uses pygame for gamepad/joystick support.

The input manager provides smooth, precise controls suitable for
collecting high-quality training data.
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseInputManager(ABC):
    """Abstract base class for input backends."""

    def __init__(self, smoothing: float = 0.3):
        """
        Args:
            smoothing: Interpolation factor for smooth transitions (0-1).
                       Lower = smoother but more lag.
                       Set to 1.0 for instant/digital response.
        """
        self.smoothing = smoothing
        self._steering = 0.0
        self._throttle = 0.0
        self._running = False

    @abstractmethod
    def start(self):
        """Start capturing input."""
        pass

    @abstractmethod
    def stop(self):
        """Stop capturing input."""
        pass

    @abstractmethod
    def _update_raw_inputs(self) -> tuple:
        """
        Return raw target values from the input device.

        Returns:
            (target_steering, target_throttle) each in [-1, 1].
        """
        pass

    def get_action(self) -> np.ndarray:
        """
        Get the current action with smoothing applied.

        Returns:
            numpy array [steering, throttle] each in [-1, 1].
        """
        target_steering, target_throttle = self._update_raw_inputs()

        self._steering += (target_steering - self._steering) * self.smoothing
        self._throttle += (target_throttle - self._throttle) * self.smoothing

        # Deadzone: snap to 0 if very close
        if abs(self._steering) < 0.01:
            self._steering = 0.0
        if abs(self._throttle) < 0.01:
            self._throttle = 0.0

        return np.array([self._steering, self._throttle], dtype=np.float32)

    def reset(self):
        """Reset input state to neutral."""
        self._steering = 0.0
        self._throttle = 0.0


class KeyboardInputManager(BaseInputManager):
    """
    Keyboard input using pynput for global key capture.

    Controls:
        W / Up Arrow    : Accelerate
        S / Down Arrow  : Brake / Reverse
        A / Left Arrow  : Steer left
        D / Right Arrow : Steer right
        Space           : Emergency brake
        Q               : Quit
    """

    def __init__(self, smoothing: float = 0.3):
        super().__init__(smoothing)
        self._keys_pressed = set()
        self._listener = None
        self._quit_requested = False
        self._lock = threading.Lock()

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        self._running = True
        self._quit_requested = False

        def on_press(key):
            with self._lock:
                self._keys_pressed.add(self._normalize_key(key))

        def on_release(key):
            with self._lock:
                self._keys_pressed.discard(self._normalize_key(key))

        self._listener = keyboard.Listener(
            on_press=on_press, on_release=on_release
        )
        self._listener.start()
        print("Keyboard input active. Controls:")
        print("  W/Up: Accelerate | S/Down: Brake")
        print("  A/Left: Steer left | D/Right: Steer right")
        print("  Space: Emergency brake | Q: Quit")

    def stop(self):
        """Stop the keyboard listener."""
        self._running = False
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _normalize_key(key):
        """Normalize key to a comparable string."""
        from pynput import keyboard

        if isinstance(key, keyboard.KeyCode):
            return key.char.lower() if key.char else None
        elif isinstance(key, keyboard.Key):
            return key.name
        return None

    def _update_raw_inputs(self) -> tuple:
        with self._lock:
            keys = set(self._keys_pressed)

        target_steering = 0.0
        target_throttle = 0.0

        # Steering
        left = "a" in keys or "left" in keys
        right = "d" in keys or "right" in keys
        if left and not right:
            target_steering = -1.0
        elif right and not left:
            target_steering = 1.0

        # Throttle
        up = "w" in keys or "up" in keys
        down = "s" in keys or "down" in keys
        space = "space" in keys

        if space:
            target_throttle = -1.0
        elif up and not down:
            target_throttle = 1.0
        elif down and not up:
            target_throttle = -1.0

        # Check quit
        if "q" in keys:
            self._quit_requested = True

        return target_steering, target_throttle

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested


class ControllerInputManager(BaseInputManager):
    """
    Gamepad/joystick input using pygame.

    Controls:
        Left stick X-axis  : Steering (-1 left to 1 right)
        Right trigger (RT) : Accelerate (0 to 1)
        Left trigger (LT)  : Brake (0 to -1)
        Button B / Circle  : Quit
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        joystick_index: int = 0,
        deadzone: float = 0.05,
    ):
        super().__init__(smoothing)
        self._joystick_index = joystick_index
        self._deadzone = deadzone
        self._joystick = None
        self._quit_requested = False

    def start(self):
        """Initialize pygame and the joystick."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError(
                "No controller detected. Connect a gamepad and retry."
            )

        self._joystick = pygame.joystick.Joystick(self._joystick_index)
        self._joystick.init()
        self._running = True

        print(f"Controller connected: {self._joystick.get_name()}")
        print("  Left stick: Steering | RT: Accelerate | LT: Brake")
        print(f"  Axes: {self._joystick.get_numaxes()}")
        print(f"  Buttons: {self._joystick.get_numbuttons()}")

    def stop(self):
        """Stop and clean up pygame."""
        import pygame

        self._running = False
        if self._joystick is not None:
            self._joystick.quit()
            self._joystick = None
        pygame.quit()

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to analog input."""
        if abs(value) < self._deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self._deadzone) / (1.0 - self._deadzone)

    def _update_raw_inputs(self) -> tuple:
        import pygame

        pygame.event.pump()

        # Left stick X-axis for steering (axis 0)
        steering = self._apply_deadzone(self._joystick.get_axis(0))

        # Triggers for throttle
        # On most controllers:
        # Axis 4 = Left trigger (LT), Axis 5 = Right trigger (RT)
        # Triggers go from -1 (released) to 1 (fully pressed)
        num_axes = self._joystick.get_numaxes()

        throttle = 0.0
        if num_axes >= 6:
            # Xbox-style controller
            rt = (self._joystick.get_axis(5) + 1.0) / 2.0  # 0 to 1
            lt = (self._joystick.get_axis(4) + 1.0) / 2.0  # 0 to 1
            throttle = rt - lt
        elif num_axes >= 4:
            # Fallback: use axis 1 (left stick Y) inverted
            throttle = -self._apply_deadzone(self._joystick.get_axis(1))

        # Quit button (B on Xbox = button 1, Circle on PS = button 2)
        for btn_idx in [1, 2]:
            if btn_idx < self._joystick.get_numbuttons():
                if self._joystick.get_button(btn_idx):
                    self._quit_requested = True

        return steering, throttle

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested


def create_input_manager(
    mode: str = "keyboard",
    smoothing: Optional[float] = None,
    **kwargs,
) -> BaseInputManager:
    """
    Factory function to create an input manager.

    Args:
        mode: "keyboard" or "controller".
        smoothing: Interpolation factor. Default depends on mode.
        **kwargs: Additional arguments for the specific backend.

    Returns:
        An InputManager instance.
    """
    if smoothing is None:
        smoothing = 0.3 if mode == "keyboard" else 0.1

    if mode == "keyboard":
        return KeyboardInputManager(smoothing=smoothing)
    elif mode == "controller":
        try:
            import pygame  # noqa: F401
        except ImportError:
            print("Error: pygame is required for controller support.")
            print("Install it with: pip install pygame")
            print("(requires SDL2 dev libraries: sudo apt install libsdl2-dev)")
            raise SystemExit(1)
        return ControllerInputManager(smoothing=smoothing, **kwargs)
    else:
        raise ValueError(f"Unknown input mode: {mode}. Use 'keyboard' or 'controller'.")
