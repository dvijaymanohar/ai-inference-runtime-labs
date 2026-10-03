from pathlib import Path
import onnxruntime as ort
import numpy as np

path=Path("models/tiny_net.onnx")
if not path.exists(): raise SystemExit("Run examples/export_and_compare.py first")
available=ort.get_available_providers()
preferred=["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]
providers=[p for p in preferred if p in available]
session=ort.InferenceSession(str(path),providers=providers)
print("requested:",providers)
print("session providers:",session.get_providers())
x=np.zeros((4,16),dtype=np.float32)
y=session.run(None,{"input":x})[0]
print("output shape:",y.shape)
print("Next: enable ORT profiling/logging and verify actual node placement. A provider being listed does not prove every operator ran there.")
