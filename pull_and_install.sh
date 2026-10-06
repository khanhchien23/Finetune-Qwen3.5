#!/bin/bash
# ===== "script nhỏ: pull source + cài thư viện" trong sơ đồ =====
# Dùng conda cho TOÀN BỘ (Python + CUDA Toolkit/nvcc + thư viện Python).
# Không cần sudo/apt-get, không phụ thuộc bản Linux/Ubuntu của máy đang chạy -
# tránh đúng loại lỗi "không tìm thấy package" gặp phải khi dùng apt trên các
# máy/image khác nhau.
set -e

REPO_URL="https://github.com/khanhchien23/Finetune-Qwen3.5.git"
SOURCE_DIR=~/source_code
CONDA_DIR=~/miniconda3
ENV_NAME="qwen_env"
PY_VERSION="3.10"
CUDA_VERSION="12.8.0"   # phải khớp bản torch==2.8.0+cu128 cài bên dưới

# ---------------------------------------------------------------------
# 0) Cài Miniconda nếu chưa có - cài gọn trong $HOME, KHÔNG cần sudo
# ---------------------------------------------------------------------
if [ ! -d "$CONDA_DIR" ]; then
    echo ">> Cài Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Dùng hẳn kênh conda-forge (cộng đồng, miễn phí), KHÔNG dùng kênh "defaults"
# của Anaconda - kênh đó từ 2024 yêu cầu chấp nhận Terms of Service thủ công
# trước khi dùng, gây lỗi "CondaToSNonInteractiveError" trên máy mới chưa
# từng accept. conda-forge không bị ràng buộc này.
conda config --add channels conda-forge
conda config --set channel_priority strict

# ---------------------------------------------------------------------
# 1) Pull code mới nhất từ GitHub
# ---------------------------------------------------------------------
if [ -d "$SOURCE_DIR/.git" ]; then
    echo ">> Đã có repo, pull bản mới nhất..."
    git -C "$SOURCE_DIR" pull
else
    echo ">> Clone repo lần đầu..."
    git clone "$REPO_URL" "$SOURCE_DIR"
fi

# ---------------------------------------------------------------------
# 2) Tạo + kích hoạt conda environment (chỉ tạo nếu chưa có)
# ---------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo ">> Tạo conda environment lần đầu (Python ${PY_VERSION})..."
    conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=${PY_VERSION}"
fi
conda activate "$ENV_NAME"
echo ">> Đang dùng: $(python --version) (conda env: $ENV_NAME)"

# ---------------------------------------------------------------------
# 3) Cài CUDA Toolkit (gồm nvcc) + thư viện Python - CHỈ chạy nếu chưa cài
#    (đánh dấu bằng file .deps_installed, nằm trong chính thư mục env)
# ---------------------------------------------------------------------
MARKER="$CONDA_DIR/envs/$ENV_NAME/.deps_installed"
if [ ! -f "$MARKER" ]; then
    echo ">> Cài CUDA Toolkit ${CUDA_VERSION} qua conda (có nvcc, không cần apt)..."
    conda install -y --override-channels -c "nvidia/label/cuda-${CUDA_VERSION}" -c conda-forge cuda-toolkit
    echo ">> nvcc: $(nvcc --version | tail -1)"

    echo ">> Cài rclone qua conda (để mount Google Drive, không cần apt)..."
    conda install -y --override-channels -c conda-forge rclone
    echo ">> rclone: $(rclone version | head -1)"

    echo ">> Cài thư viện Python (sẽ mất vài phút)..."
    pip install --upgrade -qqq pip uv

    uv pip install -qqq \
        "torch==2.8.0" "triton>=3.3.0" numpy pillow torchvision bitsandbytes xformers==0.0.32.post2 \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
    uv pip install -qqq --no-deps "torchcodec==0.7.0"
    uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
    uv pip install transformers==5.2.0
    uv pip uninstall -qqq flash-linear-attention fla-core || true
    uv pip install --no-build-isolation causal_conv1d==1.6.0
    uv pip install --no-deps --upgrade "torchao>=0.16.0"

    uv pip install huggingface_hub wandb datasets

    if [ -f "$SOURCE_DIR/requirements.txt" ]; then
        uv pip install -r "$SOURCE_DIR/requirements.txt"
    fi

    touch "$MARKER"
    echo ">> Cài thư viện xong."
else
    echo ">> Thư viện đã cài từ trước, bỏ qua."
fi

echo ">> pull_and_install.sh xong. Code ở: $SOURCE_DIR | conda env: $ENV_NAME"
