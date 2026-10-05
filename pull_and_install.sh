#!/bin/bash
# ===== "script nhỏ: pull source + cài thư viện" trong sơ đồ =====
# Chỉ lo 2 việc: (1) lấy code mới nhất từ GitHub, (2) cài đúng thư viện cần cho train.
# Không mount Drive, không chạy train - đó là việc của run.sh (gọi script này trước).
set -e

REPO_URL="https://github.com/khanhchien23/Finetune-Qwen3.5.git"   # <-- SỬA đúng repo của bạn
SOURCE_DIR=~/source_code
VENV_DIR=~/qwen_env

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

    # Nếu repo của bạn có requirements.txt riêng, cài thêm (không bắt buộc)
    if [ -f "$SOURCE_DIR/requirements.txt" ]; then
        uv pip install -r "$SOURCE_DIR/requirements.txt"
    fi

    touch "$MARKER"
    echo ">> Cài thư viện xong."
else
    echo ">> Thư viện đã cài từ trước, bỏ qua."
fi

echo ">> pull_and_install.sh xong. Code ở: $SOURCE_DIR | venv: $VENV_DIR"
