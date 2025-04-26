#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines return the exit status of the last command to exit with a non-zero status,
# or zero if no command exited with a non-zero status.
set -o pipefail

# --- Configuration ---
MINICONDA_INSTALL_PATH="$HOME/miniconda3"
CONDA_ENV_NAME="sglang_env" # Changed environment name
PYTHON_VERSION="3.11" # SGLang/vLLM often work well with 3.10/3.11
PYTORCH_CUDA_VERSION="12.1" # IMPORTANT: Match this with your installed CUDA Toolkit version
# Pin PyTorch version known to be compatible or target the one needed by torchaudio if that's critical
# Let's try pinning to 2.3.1, often stable, or use 2.5.1 if torchaudio conflict is primary concern
PINNED_TORCH_VERSION="2.3.1"
PINNED_TORCHVISION_VERSION="0.18.1"
PINNED_TORCHAUDIO_VERSION="2.3.1" # Match torch version
MINICONDA_SCRIPT_NAME="Miniconda3-latest-Linux-x86_64.sh"

# SGLang Server Configuration
MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct" # User confirmed model name
TENSOR_PARALLEL_SIZE=2                   # Using 2 GPUs
MEM_FRACTION_STATIC=0.90                 # Use SGLang's arg instead of gpu-memory-utilization
MAX_RUNNING_REQUESTS=1024                # Use SGLang's arg instead of max-num-seqs (adjust if needed)
MAX_TOTAL_TOKENS=200000                  # Keep this vLLM passthrough arg, hopefully still works
HOST_IP="0.0.0.0"
PORT="30000"                             # Changed port for SGLang server
LOG_LEVEL="info"                         # Log level (info or debug)

# --- 1. System Updates and Dependencies ---
echo "Updating package lists and installing dependencies (wget, btop)..."
# Assuming running as root (like root@...), 'sudo' is removed.
apt update
apt install -y wget btop build-essential python3-pip git git-lfs # Added git and git-lfs
git lfs install
echo "System dependencies installed."

# --- 2. Download and Install Miniconda (Non-Interactive) ---
# (No changes here)
echo "Downloading Miniconda installer..."
if [ -f "$MINICONDA_SCRIPT_NAME" ]; then
    echo "Miniconda installer already downloaded."
else
    wget https://repo.anaconda.com/miniconda/$MINICONDA_SCRIPT_NAME -O $MINICONDA_SCRIPT_NAME
fi

echo "Installing Miniconda to $MINICONDA_INSTALL_PATH..."
if [ -d "$MINICONDA_INSTALL_PATH" ]; then
    echo "Miniconda directory already exists. Skipping installation."
else
    bash $MINICONDA_SCRIPT_NAME -b -p $MINICONDA_INSTALL_PATH
    echo "Miniconda installed."
    echo "Cleaning up Miniconda installer script..."
    rm $MINICONDA_SCRIPT_NAME
fi

# --- 3. Initialize Conda for this script ---
# (No changes here)
echo "Initializing Conda environment for script..."
eval "$($MINICONDA_INSTALL_PATH/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
eval "$($MINICONDA_INSTALL_PATH/bin/conda shell.bash hook)"

# --- 4. Create and Setup Conda Environment ---
# (No changes here, uses CONDA_ENV_NAME="sglang_env")
echo "Creating Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
if conda info --envs | grep -q "^$CONDA_ENV_NAME\s"; then
   echo "Conda environment '$CONDA_ENV_NAME' already exists. Activating and ensuring correct packages."
   conda activate $CONDA_ENV_NAME
else
   conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
   echo "Conda environment '$CONDA_ENV_NAME' created. Activating..."
   conda activate $CONDA_ENV_NAME
   echo "Conda environment '$CONDA_ENV_NAME' activated."
fi

# --- 5. Install PyTorch, SGLang and other Python packages ---
echo "Installing/Updating PyTorch $PINNED_TORCH_VERSION for CUDA $PYTORCH_CUDA_VERSION..."
# Install PINNED PyTorch version FIRST to try and avoid conflicts
conda run -n $CONDA_ENV_NAME pip install --upgrade pip
conda run -n $CONDA_ENV_NAME pip install "torch==$PINNED_TORCH_VERSION" "torchvision==$PINNED_TORCHVISION_VERSION" "torchaudio==$PINNED_TORCHAUDIO_VERSION" --index-url "https://download.pytorch.org/whl/cu${PYTORCH_CUDA_VERSION//./}"

echo "Installing/Updating SGLang (with Server Runtime) and nvitop into the '$CONDA_ENV_NAME' environment..."
# Install sglang[srt] which includes vLLM dependencies
# This should now respect the already installed torch version
conda run -n $CONDA_ENV_NAME pip install "sglang[srt]" nvitop Pillow

# Verify torch version after sglang install
echo "Verifying torch version after sglang installation..."
conda run -n $CONDA_ENV_NAME python -c "import torch; print(f'Torch version: {torch.__version__}')" || echo "Verification failed"

echo "PyTorch, SGLang, and other packages installed/updated successfully in '$CONDA_ENV_NAME'."

# --- 6. Start SGLang Server (Using conda activate for safety) ---
echo "--------------------------------------------------"
echo "Installation complete. Starting SGLang API server..."
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Host: $HOST_IP"
echo "Port: $PORT"
echo "Max Running Requests: $MAX_RUNNING_REQUESTS" # Updated label
echo "Static Memory Fraction: $MEM_FRACTION_STATIC" # Updated label
echo "Max Total Tokens: $MAX_TOTAL_TOKENS"
echo "Press Ctrl+C to stop the server."
echo "--------------------------------------------------"

# Ensure environment is active before launching
if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" && "$CONDA_PREFIX" != "$MINICONDA_INSTALL_PATH/envs/$CONDA_ENV_NAME" ]]; then
    echo "Attempting to activate conda environment '$CONDA_ENV_NAME'..."
    conda activate $CONDA_ENV_NAME
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" && "$CONDA_PREFIX" != "$MINICONDA_INSTALL_PATH/envs/$CONDA_ENV_NAME" ]]; then
      echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'. Exiting."
      conda info --envs || echo "Failed to get conda info."
      exit 1
    fi
fi
echo "Conda environment '$CONDA_DEFAULT_ENV' is active."

# Launch the SGLang server with MODIFIED arguments based on usage output
echo "Launching SGLang server with updated arguments..."
python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --tokenizer-path "$MODEL_NAME" \
    --tp $TENSOR_PARALLEL_SIZE \
    --host "$HOST_IP" \
    --port $PORT \
    --dtype bfloat16 \
    --mem-fraction-static $MEM_FRACTION_STATIC \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --log-level "$LOG_LEVEL"

echo "SGLang server process finished or was stopped."

conda deactivate || echo "Note: conda deactivate finished (ignore errors if any)."

echo "Script finished."

exit 0
