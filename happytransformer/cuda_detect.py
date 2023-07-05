import torch

def detect_cuda_device_number():

    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return torch.device("mps")
        # todo
        return -1

    return torch.cuda.current_device() if torch.cuda.is_available() else -1