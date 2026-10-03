import statistics, time, torch
from torch import nn

device="cuda" if torch.cuda.is_available() else "cpu"
model=nn.Sequential(nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,1024)).to(device).eval()

def sync():
    if device=="cuda": torch.cuda.synchronize()

@torch.inference_mode()
def measure(batch):
    x=torch.randn(batch,1024,device=device)
    for _ in range(10): model(x)
    sync(); samples=[]
    for _ in range(30):
        t=time.perf_counter(); model(x); sync(); samples.append((time.perf_counter()-t)*1000)
    med=statistics.median(samples)
    return {"batch":batch,"median_ms":med,"items_per_s":batch/(med/1000)}

for b in [1,2,4,8,16,32,64,128]:
    print(measure(b))
