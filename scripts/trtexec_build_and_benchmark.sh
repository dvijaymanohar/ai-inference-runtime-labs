#!/usr/bin/env bash
set -euo pipefail
ONNX="${1:-models/tiny_net.onnx}"
ENGINE="${2:-models/tiny_net.engine}"
if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install a TensorRT version compatible with your CUDA/driver stack." >&2
  exit 77
fi
echo "== Build engine =="
trtexec --onnx="$ONNX" --saveEngine="$ENGINE" --skipInference
echo "== Benchmark steady state =="
trtexec --loadEngine="$ENGINE" --warmUp=1000 --duration=10 --useCudaGraph
echo "Record GPU, driver, CUDA, TensorRT, shape profile, precision, and command line with any result."
