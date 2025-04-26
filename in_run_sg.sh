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
PYTHON_VERSION="3.11" # SGLang/vLLM often work well with 3.10/3.11, 3.12 might be too new sometimes, adjust if needed
PYTORCH_CUDA_VERSION="12.1" # IMPORTANT: Match this with your installed CUDA Toolkit version
MINICONDA_SCRIPT_NAME="Miniconda3-latest-Linux-x86_64.sh"

# SGLang Server Configuration
MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct" # User confirmed model name
TENSOR_PARALLEL_SIZE=2                   # Using 2 GPUs
GPU_MEMORY_UTILIZATION=0.95              # Increased slightly for A100 80G
MAX_NUM_SEQS=1024                        # Increased significantly for higher concurrency on A100 80Gx2
MAX_TOTAL_TOKENS=200000                  # Optional: Limit total tokens in KV cache for stability, adjust based on workload
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
echo "Initializing Conda environment for script..."
eval "$($MINICONDA_INSTALL_PATH/bin/conda shell.bash hook)"
# Ensure conda base environment is not activated by default in subsequent shells if needed
conda config --set auto_activate_base false
# Re-initialize for current shell
eval "$($MINICONDA_INSTALL_PATH/bin/conda shell.bash hook)"
# Add conda init to bashrc if not already present, but commented out for non-interactive setup focus
# $MINICONDA_INSTALL_PATH/bin/conda init bash > /dev/null 2>&1

# --- 4. Create and Setup Conda Environment ---
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
echo "Installing/Updating PyTorch for CUDA $PYTORCH_CUDA_VERSION..."
# Install PyTorch first to ensure compatibility
conda run -n $CONDA_ENV_NAME pip install --upgrade pip
conda run -n $CONDA_ENV_NAME pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu${PYTORCH_CUDA_VERSION//./}" # Dynamically build URL

echo "Installing/Updating SGLang (with Server Runtime) and nvitop into the '$CONDA_ENV_NAME' environment..."
# Install sglang[srt] which includes vLLM dependencies
conda run -n $CONDA_ENV_NAME pip install "sglang[srt]" nvitop Pillow # Added Pillow for image handling

echo "PyTorch, SGLang, and other packages installed/updated successfully in '$CONDA_ENV_NAME'."

# --- 6. Start SGLang Server (Using conda activate for safety) ---
echo "--------------------------------------------------"
echo "Installation complete. Starting SGLang API server..."
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Host: $HOST_IP"
echo "Port: $PORT"
echo "Max Sequences: $MAX_NUM_SEQS"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Press Ctrl+C to stop the server."
echo "--------------------------------------------------"

# Ensure environment is active before launching
if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" && "$CONDA_PREFIX" != "$MINICONDA_INSTALL_PATH/envs/$CONDA_ENV_NAME" ]]; then
    echo "Attempting to activate conda environment '$CONDA_ENV_NAME'..."
    conda activate $CONDA_ENV_NAME
    # Double check activation
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" && "$CONDA_PREFIX" != "$MINICONDA_INSTALL_PATH/envs/$CONDA_ENV_NAME" ]]; then
      echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'. Exiting."
      conda info --envs || echo "Failed to get conda info."
      exit 1
    fi
fi
echo "Conda environment '$CONDA_DEFAULT_ENV' is active."

# Launch the SGLang server
# Note: --trust-remote-code is often implicitly handled by HuggingFace loaders used by vLLM/SGLang.
# If you encounter issues related to custom code, check if the model requires explicit trust.
python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --tokenizer-path "$MODEL_NAME" \
    --tp $TENSOR_PARALLEL_SIZE \
    --host "$HOST_IP" \
    --port $PORT \
    --dtype bfloat16 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --log-level "$LOG_LEVEL" \
    --disable-log-stats # Added to potentially improve performance

echo "SGLang server process finished or was stopped."

# Deactivate environment - This might not run if the server is killed with Ctrl+C directly
# but it's good practice in a script that might continue.
conda deactivate || echo "Note: conda deactivate finished (ignore errors if any)."

echo "Script finished."

exit 0
