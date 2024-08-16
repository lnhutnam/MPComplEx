import os

import random
import numpy as np

import torch


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.backends.cudnn.benchmark = True
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def com_mult(a, b):
    return torch.mul(a, b) + a + b


def com_add(a, b):
    return torch.add(a, b)


def com_sub(a, b):
    return torch.subtract(a, b)


def com_corr(a, b):
    return (
        torch.fft.irfftn(
            torch.conj(torch.fft.rfftn(a, (-1))) * torch.fft.rfftn(b, (-1)), (-1)
        )
        + a
        + b
    )


def poly_ker2(x, d=2):
    return (1 + x * x) ** d


def poly_ker3(x, d=3):
    return (1 + x * x) ** d


def poly_ker5(x, d=5):
    return (1 + x * x) ** d


def gauss_ker(x, z=0, sigma=1):
    expo = -abs(x - z) ** 2 / (2 * sigma**2)
    return torch.exp(expo)
