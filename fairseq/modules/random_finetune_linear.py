import math
import torch 
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class RFTLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, prob: float = 0.005, mask_type: int = 1) -> None:
        super(RFTLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = prob
        self.mask_type = mask_type

        if self.mask_type == 1:
            self.th_w = torch.rand([in_features], requires_grad=False).cuda()
            self.th_b = torch.rand([1], requires_grad=False).cuda()
        elif self.mask_type == 2:
            self.th_w = torch.rand([out_features, in_features], requires_grad=False).cuda()
            self.th_b = torch.rand([out_features], requires_grad=False).cuda()

        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.weight_upd = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features), requires_grad=False)
            self.bias_upd = Parameter(torch.Tensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_upd, a=math.sqrt(5))

        if self.bias_upd is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_upd)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_upd, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight * (self.th_w > self.p) + self.weight_upd * (self.th_w <= self.p)
        bias = self.bias * (self.th_b > self.p) + self.bias_upd * (self.th_b <= self.p)
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ft_rate={}'.format(
            self.in_features, self.out_features, self.bias_upd is not None, self.p
        )

class NogradLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NogradLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

