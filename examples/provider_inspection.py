import onnxruntime as ort
print("available_providers:", ort.get_available_providers())
print("device:", ort.get_device())
print("Exercise: install a GPU-enabled runtime, repeat, then inspect actual provider placement rather than assuming CUDA execution.")
