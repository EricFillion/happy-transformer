import torch
import torch_xla
import torch_xla.core.xla_model as xm

def detect_cuda_device_number():
    return torch.cuda.current_device() if torch.cuda.is_available() else -1

def detect_tpu_device_number():
    return xm.xla_device().index if xm.xla_device() else -1
