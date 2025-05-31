from typing import List
import torch
import torch._dynamo

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * 1
    return x * b


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

compiled_fn = torch.compile(toy_example, backend = my_compiler)
print(compiled_fn(torch.randn(4), torch.randn(4)))


onnx_model = torch.onnx.dynamo_export(
    toy_example,
    (torch.randn(4), torch.randn(4))
)
import pdb; pdb.set_trace()
