# AI Inference Runtime Labs

Compare model inference across PyTorch and ONNX Runtime, then extend to TensorRT when supported by the local NVIDIA environment.

## Sequence
PyTorch baseline → inference_mode → batch sweep → precision sweep → ONNX export → cross-runtime validation → ONNX Runtime providers/fallback → TensorRT/trtexec → dynamic shapes → quantization → end-to-end vs device-only timing.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/export_and_compare.py
pytest -q
```

Install a GPU-specific ONNX Runtime/TensorRT stack separately when available; dependency compatibility is platform/version sensitive.
