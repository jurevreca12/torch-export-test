import torch

def foo(x, y):
    return 2 * x + y

# only tensor ops (no dynamic control flow)
traced_foo = torch.jit.trace(
    foo,
    (torch.rand(3), torch.rand(3))
)

# also supports dynamic control flow 
script_foo = torch.jit.script(
    foo,
    (torch.rand(3), torch.rand(3))
)
import pdb; pdb.set_trace()
