import torch
from torch.export import export, Dim

@torch.library.custom_op("mylib::preproc", mutates_args={})
def preproc_impl(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / t.std()

@torch.library.register_fake("mylib::preproc")
def preproc_fake(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / t.std()


class M(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.ops.mylib.preproc(x)
        #assert x.shape[0] == 6
        a = y.item()
        torch._check_is_size(a)
        a = a // 3
        a = a + 5
        z = torch.cat([x, x])
        torch._check(a < z.shape[0])
        return z[:a]

inp1 = (torch.randn(6), torch.tensor(2))
print(M()(*inp1))
ep = export(
    M(), 
    inp1, 
    dynamic_shapes=({0: Dim("seq_len")}, None)
)
import pdb; pdb.set_trace()
