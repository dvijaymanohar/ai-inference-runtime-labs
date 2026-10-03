import statistics, time
import numpy as np
import onnxruntime as ort
import torch
from examples.export_and_compare import TinyNet

def bench(fn,warm=10,reps=50):
    for _ in range(warm): fn()
    xs=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); xs.append((time.perf_counter()-t)*1000)
    return statistics.median(xs),min(xs),max(xs)

torch.manual_seed(0); model=TinyNet().eval(); x=torch.randn(32,16)
with torch.inference_mode():
    pt=bench(lambda:model(x))
sess=ort.InferenceSession("models/tiny_net.onnx",providers=["CPUExecutionProvider"])
arr=x.numpy(); ort_stats=bench(lambda:sess.run(None,{"input":arr}))
print({"pytorch_ms":pt,"onnxruntime_ms":ort_stats,"boundary":"host end-to-end call"})
