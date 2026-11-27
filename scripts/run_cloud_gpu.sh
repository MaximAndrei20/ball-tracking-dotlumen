#!/usr/bin/env bash
# Runs Stage 1 and Stage 3 on a Linux CUDA GPU box and packs cache/gpu/ for local sync.
set -euo pipefail

# Usage on a CUDA machine:
#   DETECTOR=sam2 DA3_REPO=/workspace/Depth-Anything-3 bash scripts/run_cloud_gpu.sh

echo "=== Cloud GPU Pre-Flight Check ==="
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch, ultralytics; assert torch.cuda.is_available(), 'CUDA unavailable'; print('PyTorch CUDA OK:', torch.cuda.get_device_name(0), '| Ultralytics:', ultralytics.__version__)"

DETECTOR="${DETECTOR:-sam2}"
DA3_REPO="${DA3_REPO:-}"
DA3_CFG="${DA3_CFG:-configs/base_config.yaml}"

python3 run_pipeline.py --profile gpu --stage none --export-frames
python3 run_pipeline.py --profile gpu --stage 1 --detector "${DETECTOR}"
python3 run_pipeline.py --profile gpu --stage 2

FRAMES_DIR="$(realpath cache/gpu/frames)"
DA3_OUT="$(realpath -m cache/gpu/da3)"
mkdir -p "${DA3_OUT}"

if [ -n "${DA3_REPO}" ] && [ -d "${DA3_REPO}" ]; then
  DA3_ABS="$(realpath "${DA3_REPO}")"
  if [ -f "${DA3_ABS}/da3_streaming/da3_streaming.py" ]; then
    (cd "${DA3_ABS}/da3_streaming" && python3 da3_streaming.py --image_dir "${FRAMES_DIR}" --config "${DA3_CFG}" --output_dir "${DA3_OUT}")
  elif [ -f "${DA3_ABS}/da3_streaming.py" ]; then
    (cd "${DA3_ABS}" && python3 da3_streaming.py --image_dir "${FRAMES_DIR}" --config "${DA3_CFG}" --output_dir "${DA3_OUT}")
  else
    echo "ERROR: Could not locate da3_streaming.py inside '${DA3_ABS}'."
    exit 1
  fi
else
  echo "Set DA3_REPO=/path/to/Depth-Anything-3 or copy camera_poses.txt + combined_pcd.ply into '${DA3_OUT}'."
fi

python3 run_pipeline.py --profile gpu --stage 3 --vo da3
tar -czf cache_gpu_sync.tar.gz cache/gpu/stage1_2d.npz cache/gpu/stage1_2d.csv cache/gpu/stage3_ego.npz
echo "Packed cache_gpu_sync.tar.gz (~5-10 MB). Extract into d:/Facultate/dotlumen/ and run:"
echo "  .venv\\Scripts\\python.exe run_pipeline.py --profile gpu --stage 2,4,5 --compare-profiles"
