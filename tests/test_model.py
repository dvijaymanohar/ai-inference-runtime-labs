import torch
from examples.export_and_compare import TinyNet

def test_shape_and_determinism():
    torch.manual_seed(123)
    m=TinyNet().eval(); x=torch.randn(3,16)
    with torch.inference_mode():
        a=m(x); b=m(x)
    assert a.shape==(3,8)
    torch.testing.assert_close(a,b)
