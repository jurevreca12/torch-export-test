import copy
import torch
from torchao.quantization import quantize_
from torchao.quantization.qat import FakeQuantizeConfig, IntXQuantizationAwareTrainingConfig



class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

float_model = MyModel()
example_inputs = torch.ones(1, 10)

activation_config = FakeQuantizeConfig(
   torch.int8, "per_token", is_symmetric=False,
)
weight_config = FakeQuantizeConfig(
   torch.int4, group_size=10, is_symmetric=True,
)
quant_model = copy.deepcopy(float_model)
quantize_(
    quant_model,
    IntXQuantizationAwareTrainingConfig(activation_config, weight_config),
)
import pdb; pdb.set_trace()
