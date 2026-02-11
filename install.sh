#!/bin/bash
# Installation script for Racing Simulator AI
# Handles mlagents-envs which requires Python 3.10 officially
# but works on 3.11+ when installed from source with relaxed constraints.

set -e

echo "=== Racing Simulator AI - Installation ==="
echo ""

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Install standard dependencies
echo ""
echo "[1/2] Installing Python dependencies..."
pip install -r requirements.txt

# Install mlagents-envs from GitHub (bypassing Python version constraint)
echo ""
echo "[2/2] Installing mlagents-envs from GitHub source..."
echo "      (bypassing strict Python version constraint for Python $PYTHON_VERSION)"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

git clone --depth 1 --branch develop https://github.com/Unity-Technologies/ml-agents.git "$TMPDIR/ml-agents"

# Relax Python version constraint (officially >=3.10.1,<=3.10.12)
sed -i 's/python_requires=">=3.10.1,<=3.10.12"/python_requires=">=3.10.1"/' "$TMPDIR/ml-agents/ml-agents-envs/setup.py"

# Relax numpy constraint for Python 3.14 compatibility
sed -i 's/numpy>=1.23.5,<1.24.0/numpy>=1.23.0/' "$TMPDIR/ml-agents/ml-agents-envs/setup.py"

# Relax pettingzoo pin
sed -i 's/pettingzoo==1.15.0/pettingzoo>=1.15.0/' "$TMPDIR/ml-agents/ml-agents-envs/setup.py"

# Install with --no-deps since we already installed compatible deps
pip install "$TMPDIR/ml-agents/ml-agents-envs" --no-deps

echo ""
echo "=== Installation complete ==="
echo ""
echo "Verify with:"
echo "  python3 -c 'from mlagents_envs.environment import UnityEnvironment; print(\"mlagents_envs OK\")'"
echo ""
echo "Quick start:"
echo "  python main.py collect                  # Drive and collect data"
echo "  python main.py eda                      # Analyze data quality"
echo "  python main.py train                    # Train AI model"
echo "  python main.py drive                    # AI drives autonomously"
