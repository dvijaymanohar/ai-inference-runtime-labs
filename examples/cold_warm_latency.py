import statistics, time
import onnxruntime as ort
import numpy as np

path="models/tiny_net.onnx"
t0=time.perf_counter(); session=ort.InferenceSession(path,providers=["CPUExecutionProvider"])
load_ms=(time.perf_counter()-t0)*1000
x=np.zeros((1,16),dtype=np.float32)
t0=time.perf_counter(); session.run(None,{"input":x}); first_ms=(time.perf_counter()-t0)*1000
samples=[]
for _ in range(50):
    a=time.perf_counter(); session.run(None,{"input":x}); samples.append((time.perf_counter()-a)*1000)
print({"session_load_ms":load_ms,"first_inference_ms":first_ms,
       "steady_median_ms":statistics.median(samples),"steady_min_ms":min(samples),"steady_max_ms":max(samples)})
