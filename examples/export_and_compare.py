from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,8))
    def forward(self,x): return self.net(x)

torch.manual_seed(0)
model=TinyNet().eval()
x=torch.randn(4,16)
Path("models").mkdir(exist_ok=True)
path="models/tiny_net.onnx"
with torch.inference_mode():
    ref=model(x).numpy()
torch.onnx.export(model,x,path,input_names=["input"],output_names=["output"],
                  dynamic_axes={"input":{0:"batch"},"output":{0:"batch"}},opset_version=17)
session=ort.InferenceSession(path,providers=["CPUExecutionProvider"])
got=session.run(["output"],{"input":x.numpy()})[0]
np.testing.assert_allclose(got,ref,rtol=1e-4,atol=1e-5)
print("PASS providers=",session.get_providers(),"max_abs_error=",float(np.max(np.abs(got-ref))))
