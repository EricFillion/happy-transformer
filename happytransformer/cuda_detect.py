import torch

def detect_cuda_device_number():
    return torch.cuda.current_device() if torch.cuda.is_available() else -1