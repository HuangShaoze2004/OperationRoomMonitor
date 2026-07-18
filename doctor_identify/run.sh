#!/usr/bin/env bash
# 医生身份识别系统 - 启动脚本
# 自动设置 CUDA/cuDNN 库路径，然后运行对应的 Python 脚本。
#
# 用法:
#   bash run.sh doctor_manager.py build --dataset data/people_datasets
#   bash run.sh evaluate.py --dataset data/people_datasets
#   bash run.sh identify.py --video path/to/test.mp4
#   bash run.sh run_pipeline.py --evaluate

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# CUDA/cuDNN 库路径
CONDA_ENV="$(dirname "$(dirname "$(which python)")")"
SITE_PACKAGES="$CONDA_ENV/lib/python3.12/site-packages"

export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cu13/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# 运行 Python 脚本
exec python "$@"
