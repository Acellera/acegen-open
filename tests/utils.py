import torch
from rdkit import RDLogger
import warnings

def disable_warnings(*args):
    for warning in args:
        warnings.filterwarnings("ignore", category=warning) 

def disable_rdkit_logging():
    RDLogger.DisableLog('rdApp.*')

def get_default_devices():
    num_cuda = torch.cuda.device_count()
    if num_cuda == 0:
        return [torch.device("cpu")]
    else:
        return [torch.device("cuda:0")]
