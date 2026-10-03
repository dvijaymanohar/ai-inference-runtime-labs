# TensorRT extension

Hardware/toolchain dependent.

Use a compatible TensorRT release and `trtexec` to study engine build, tactics, dynamic shapes, workspace/memory, precision, warm-up, and throughput/latency.

Example workflow:
```bash
trtexec --onnx=models/model.onnx --saveEngine=models/model.engine
trtexec --loadEngine=models/model.engine --warmUp=1000 --duration=10
```

Record TensorRT/CUDA/driver/GPU versions and validate outputs/quality before interpreting speedups. Do not compare TensorRT numbers against a differently shaped or differently measured baseline.
