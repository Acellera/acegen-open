import torch


def get_default_devices():
    num_cuda = torch.cuda.device_count()
    if num_cuda == 0:
        return [torch.device("cpu")]
    else:
        return [torch.device("cuda:0")]
