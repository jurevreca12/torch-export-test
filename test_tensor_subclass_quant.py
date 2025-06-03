import torch
import copy
from typing import Tuple, List, Any
from test_module_replace_quant import ToyModel, int8_symmetric_quantize


class Int8SymmetricTensor(torch.Tensor):
    """
    Our subclass represents a tensor that has been quantized to int8
    It will hold two inner tensors:
      int_data: int8[M, N]
      scale: fp32[M, 1]
    """

    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: torch.Tensor, scale: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            strides=int_data.stride(),
            storage_offset=int_data.storage_offset(),
            dtype=scale.dtype,
            device=int_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: torch.Tensor, scale: torch.Tensor):
        # inner data expected to be quantized already
        assert int_data.dtype is torch.int8
        # we could do more work to support ndim > 2!
        assert int_data.ndim == 2
        assert scale.ndim == 2
        self.int_data = int_data
        self.scale = scale

    def __tensor_flatten__(self) -> Tuple[List[str], Any]:
        """
        Returns a tuple of:
          names of all inner tensor attributes (two in our case)
          any other additional, non-tensor metadata.

        Needed for PT2 support.
        """
        return ["int_data", "scale"], None

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, extra_metadata, outer_size=None, outer_stride=None):
        """
         __tensor_unflatten__ should effectively undo __tensor_flatten__.

        inputs:
          a dict mapping names of inner tensor attributes back to the tensors
          the constant metadata from __tensor_flatten__
        output:
          a new instance of your subclass

        Needed for PT2 support.
        """
        assert extra_metadata is None
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        return Int8SymmetricTensor(int_data, scale)

    def __repr__(self):
        return f'Int8SymmetricTensor(int_data={repr(self.int_data)}, scale={repr(self.scale)})'

    @staticmethod
    def from_float(float_tensor):
        """
        Actually performs the symmetric quantization.
        In our simple inference example we will quantize weights "ahead-of-time",
        although later in a training example we can quantize/dequantize
        during model execution, inside of our __torch_dispatch__

        input:
          float32 torch.Tensor
        output:
          Int8SymmetricTensor
        """
        int8_tensor, scale = int8_symmetric_quantize(float_tensor)
        return Int8SymmetricTensor(int8_tensor, scale)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        """
        Called for each ATen operator that our subclass is passed as an input to.
        We need to define our own implementation for every operator here.
        """
        if kwargs is None:
            kwargs = {}
        if func not in op_implementations_dict:
            raise AssertionError(f'Int8SymmetricTensor does not yet support op: {str(func)}')
        return op_implementations_dict[func](func, *args, **kwargs)


# Convenience function for registering our own implementation
# to every ATen operator in PyTorch
op_implementations_dict = {}
def register_op(ops: List[torch._ops.OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations_dict
        for op in ops:
            op_implementations_dict[op] = op_impl
        return op_impl

    return impl_decorator


from torch.utils._python_dispatch import return_and_correct_aliasing

@register_op([torch.ops.aten.mm.default])
def int8_mm(func, x, weight):
    assert isinstance(weight, Int8SymmetricTensor), "Int8SymmetricTensor: matmul currently only supports the weight in low precision, not the input!"
    return torch.mm(x, weight.int_data.to(x.dtype)) * weight.scale

@register_op([
    torch.ops.aten.detach.default,
    torch.ops.aten.t.default,
])
def int8_view_ops(func, *args, **kwargs):
    assert isinstance(args[0], Int8SymmetricTensor)
    out_data = func(args[0].int_data, *args[1:], **kwargs)
    out_scale = func(args[0].scale, *args[1:], **kwargs)
    out = Int8SymmetricTensor(out_data, out_scale)
    return return_and_correct_aliasing(func, args, kwargs, out)


if __name__ == '__main__':
    float_model = ToyModel(64, 128, 32)
    quantized_model_subclass = copy.deepcopy(float_model)
    for name, child in quantized_model_subclass.named_children():
        if type(child) == torch.nn.Linear:
            subclass_param = Int8SymmetricTensor.from_float(child.weight)
            child.weight = torch.nn.Parameter(subclass_param, requires_grad=True)
    with torch.no_grad():
        x = torch.randn(64, 64, 64, device='cpu')
        out = quantized_model_subclass(x)
        print(out.shape)  # prints True
        # We can also use torch.compile to fuse some of our quantized logic
        out_compiled = torch.compile(quantized_model_subclass)(x)
        print(torch.allclose(out, out_compiled))  # prints True





