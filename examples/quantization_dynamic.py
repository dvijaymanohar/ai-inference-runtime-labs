from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

src=Path("models/tiny_net.onnx")
dst=Path("models/tiny_net.int8.onnx")
if not src.exists():
    raise SystemExit("Run examples/export_and_compare.py first")
quantize_dynamic(str(src),str(dst),weight_type=QuantType.QInt8)
fp=ort.InferenceSession(str(src),providers=["CPUExecutionProvider"])
q=ort.InferenceSession(str(dst),providers=["CPUExecutionProvider"])
x=np.random.default_rng(0).standard_normal((32,16)).astype("float32")
a=fp.run(None,{"input":x})[0]
b=q.run(None,{"input":x})[0]
print("max_abs_error:",float(np.max(np.abs(a-b))))
print("mean_abs_error:",float(np.mean(np.abs(a-b))))
print("fp_model_bytes:",src.stat().st_size,"int8_model_bytes:",dst.stat().st_size)
