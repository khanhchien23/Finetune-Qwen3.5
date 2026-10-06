#!/bin/bash
# ===== "script nhỏ: pull source + cài thư viện" trong sơ đồ =====
# Chỉ lo 2 việc: (1) lấy code mới nhất từ GitHub, (2) cài đúng thư viện cần cho train.
# Không mount Drive, không chạy train - đó là việc của run.sh (gọi script này trước).
set -e

REPO_URL="https://github.com/khanhchien23/Finetune-Qwen3.5.git"
SOURCE_DIR=~/source_code
VENV_DIR=~/qwen_env
CUDA_VERSION="12.8"   # phải khớp bản torch==2.8.0+cu128 đang cài bên dưới

SUDO=""
command -v sudo &>/dev/null && SUDO="sudo"   # nhiều image Docker chạy sẵn root, không có "sudo"

# ---------------------------------------------------------------------
# 0a) Đảm bảo có Python 3.10 - cài nếu máy mới chưa có (ảnh Vast.ai có thể
#     chỉ có sẵn 3.11/3.12 mặc định, không phải lúc nào cũng có 3.10)
# ---------------------------------------------------------------------
if ! command -v python3.10 &>/dev/null; then
    echo ">> Chưa có python3.10, cài..."
    if command -v apt-get &>/dev/null; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq python3.10 python3.10-venv python3-pip
    else
        echo "!! Không tìm thấy apt-get, không tự cài được python3.10 trên hệ này."
        echo "!! Cài thủ công rồi chạy lại script."
        exit 1
    fi
fi
echo ">> python3.10: $(python3.10 --version)"

# ---------------------------------------------------------------------
# 0b) Đảm bảo có nvcc (CUDA Toolkit) - causal_conv1d cần biên dịch lúc cài,
#     nhiều máy chỉ có driver GPU (nvidia-smi chạy được) mà THIẾU trình biên
#     dịch nvcc - 2 thứ khác nhau. Nếu máy thuê dùng image CUDA "devel" sẵn
#     thì đoạn này sẽ tự bỏ qua (nvcc đã có), không tốn thời gian cài lại.
# ---------------------------------------------------------------------
if ! command -v nvcc &>/dev/null; then
    echo ">> Chưa có nvcc, cài CUDA Toolkit $CUDA_VERSION..."
    if command -v apt-get &>/dev/null; then
        cd /tmp
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        $SUDO dpkg -i cuda-keyring_1.1-1_all.deb
        $SUDO apt-get update -qq
        CUDA_PKG="cuda-toolkit-${CUDA_VERSION/./-}"   # 12.8 -> cuda-toolkit-12-8
        $SUDO apt-get install -y -qq "$CUDA_PKG" build-essential
        cd - >/dev/null

        export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH"
        # Lưu lại cho các lần mở terminal sau (không chỉ phiên script này)
        if ! grep -q "cuda-${CUDA_VERSION}/bin" ~/.bashrc 2>/dev/null; then
            echo "export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        fi
    else
        echo "!! Không tìm thấy apt-get, không tự cài được CUDA Toolkit trên hệ này."
        echo "!! Cài thủ công rồi chạy lại script."
        exit 1
    fi
fi
echo ">> nvcc: $(nvcc --version | tail -1)"

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
# 2) Tạo + kích hoạt virtual environment (chỉ tạo nếu chưa có)
# ---------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo ">> Tạo virtual environment lần đầu..."
    python3.10 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------
# 3) Cài thư viện - CHỈ chạy nếu chưa cài (đánh dấu bằng file .deps_installed)
#    Dùng đúng phiên bản đã ghim trong notebook gốc (đã test chạy được).
# ---------------------------------------------------------------------
MARKER="$VENV_DIR/.deps_installed"
if [ ! -f "$MARKER" ]; then
    echo ">> Cài thư viện lần đầu (sẽ mất vài phút)..."
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

echo ">> pull_and_install.sh xong. Code ở: $SOURCE_DIR | venv: $VENV_DIR"
