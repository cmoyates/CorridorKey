#!/bin/bash

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Ensure script stops on error
set -e

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Enable OpenEXR Support
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting Coarse Alpha Generation..."
echo "Scanning ClipsForInference..."

# Run Manager (uv handles the virtual environment automatically)
uv run python "${SCRIPT_DIR}/corridorkey_cli.py" --action generate_alphas

echo "Done."
