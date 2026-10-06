#!/bin/bash
# ===== CHẠY FILE NÀY LÀ ĐỦ - mọi thứ còn lại tự động =====
set -e

MOUNT_DIR=~/gdrive_mount
REMOTE_NAME="Chien"                               # tên remote rclone đã tạo
DRIVE_FOLDER_ID="1EPC42nUpIEpTT8YwK_V9tgZ8XgaJ47Rl"
SOURCE_DIR=~/source_code
ENV_NAME="qwen_env"

# ---------------------------------------------------------------------
# 1) Pull code + cài thư viện - dùng "source" để nhận lại CONDA_DIR
#    mà script con đã dò được (anaconda có sẵn hoặc miniconda mới cài),
#    đồng thời env đã được activate sẵn.
# ---------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/pull_and_install.sh"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---------------------------------------------------------------------
# 2) Mount Google Drive - CHỈ mount nếu chưa mount (idempotent)
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
# 3) Kiểm tra đã đăng nhập HF / wandb chưa
# ---------------------------------------------------------------------
if [ -z "$HF_TOKEN" ] && ! hf auth whoami &>/dev/null; then
    echo "!! CẢNH BÁO: chưa đăng nhập Hugging Face. Chạy: hf auth login"
    exit 1
fi
if [ -z "$WANDB_API_KEY" ] && { [ ! -f ~/.netrc ] || ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; }; then
    echo "!! CẢNH BÁO: chưa đăng nhập wandb. Chạy: wandb login"
    exit 1
fi

# ---------------------------------------------------------------------
# 4) Chạy training
# ---------------------------------------------------------------------
echo ">> Bắt đầu train..."
python "$SOURCE_DIR/train.py"