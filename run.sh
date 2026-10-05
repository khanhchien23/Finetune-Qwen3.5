#!/bin/bash
# ===== CHẠY FILE NÀY LÀ ĐỦ - mọi thứ còn lại tự động =====
set -e

VENV_DIR=~/qwen_env
MOUNT_DIR=~/gdrive_mount
REMOTE_NAME="Chien"                               # tên remote rclone đã tạo
DRIVE_FOLDER_ID="1EPC42nUpIEpTT8YwK_V9tgZ8XgaJ47Rl"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # thư mục chứa chính run.sh/train.py

# ---------------------------------------------------------------------
# 1) Tạo + kích hoạt virtual environment (chỉ tạo nếu chưa có)
# ---------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo ">> Tạo virtual environment lần đầu..."
    python3.10 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------
# 2) Cài thư viện - CHỈ chạy nếu chưa cài (đánh dấu bằng file .deps_installed)
#    Dùng đúng phiên bản đã ghim trong notebook gốc (đã test chạy được),
#    thay vì cài bản "mới nhất" dễ lệch phiên bản.
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

    touch "$MARKER"
    echo ">> Cài thư viện xong."
else
    echo ">> Thư viện đã cài từ trước, bỏ qua."
fi

# ---------------------------------------------------------------------
# 3) Mount Google Drive - CHỈ mount nếu chưa mount (idempotent)
# ---------------------------------------------------------------------
if ! mountpoint -q "$MOUNT_DIR" 2>/dev/null; then
    echo ">> Mount Google Drive..."
    mkdir -p "$MOUNT_DIR"
    rclone mount "$REMOTE_NAME": "$MOUNT_DIR" \
        --drive-root-folder-id "$DRIVE_FOLDER_ID" \
        --vfs-cache-mode full \
        --vfs-cache-max-size 20G \
        --vfs-cache-max-age 24h \
        --daemon
    sleep 3
    echo ">> Đã mount vào $MOUNT_DIR"
else
    echo ">> Google Drive đã mount sẵn, bỏ qua."
fi

# ---------------------------------------------------------------------
# 4) Kiểm tra đã đăng nhập HF / wandb chưa (chỉ cảnh báo, không tự login được
#    vì cần nhập token thủ công 1 lần duy nhất trên máy này)
# ---------------------------------------------------------------------
if [ ! -f ~/.cache/huggingface/token ] && [ -z "$HF_TOKEN" ]; then
    echo "!! CẢNH BÁO: chưa đăng nhập Hugging Face. Chạy: huggingface-cli login"
    exit 1
fi
if [ ! -f ~/.netrc ] || ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    echo "!! CẢNH BÁO: chưa đăng nhập wandb. Chạy: wandb login"
    exit 1
fi

# ---------------------------------------------------------------------
# 5) Chạy training
# ---------------------------------------------------------------------
echo ">> Bắt đầu train..."
python "$SOURCE_DIR/train.py"
