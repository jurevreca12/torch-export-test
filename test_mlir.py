from typing import List
import torch
from torch_mlir import fx

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * 1
    return x * b

print(toy_example(torch.randn(4), torch.randn(4)))

m = fx.export_and_inport(
    toy_example,
    (torch.randn(4), torch.randn(4)),
)
import pdb; pdb.set_trace()
