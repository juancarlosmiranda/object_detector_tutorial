"""
Project: Object detector and segmentation tutorial https://github.com/juancarlosmiranda/object_detector_tutorial
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2021
Description:

Adapted from
* https://www.kaggle.com/getting-started/140636
* https://stackoverflow.com/questions/71498324/pytorch-runtimeerror-cuda-out-of-memory-with-a-huge-amount-of-free-memory

Adapted by Juan Carlos Miranda as a programming practice, February 2021.

Use:
"""


import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


if __name__ == '__main__':
    print('FREE GPU MEMORY')
    free_gpu_cache()

    # pip install numba
    # pip install GPUtil
    # https://www.kaggle.com/getting-started/140636
    # https://stackoverflow.com/questions/71498324/pytorch-runtimeerror-cuda-out-of-memory-with-a-huge-amount-of-free-memory