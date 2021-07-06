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

    def __init__(self, in_features: int, out_features: int, bias: bool = True, prob: float = 0.005) -> None:
        super(RFTLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = prob

        self.th_w = torch.rand([in_features], requires_grad=False).cuda()
        self.th_b = torch.rand([1], requires_grad=False).cuda()

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
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_upd is not None
        )


# def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
#     if torch.jit.is_scripting():
#         export = True
#     if not export and torch.cuda.is_available() and has_fused_layernorm:
#         return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
#     return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)



# class RFTLayerNorm(nn.Module):
#     def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5,
#                  device=None, dtype=None, prob: float = 0.001) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(RFTLayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             # mypy error: incompatible types in assignment
#             normalized_shape = (normalized_shape,)  # type: ignore[assignment]
#         self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
#         self.eps = eps
#         self.p = prob

#         self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
#         self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
#         self.weight_upd = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
#         self.bias_upd = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))


#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         if self.elementwise_affine:
#             init.ones_(self.weight)
#             init.zeros_(self.bias)

#     def forward(self, input: Tensor) -> Tensor:
#         return F.layer_norm(
#             input, self.normalized_shape, self.weight, self.bias, self.eps)

#     def extra_repr(self) -> str:
#         return '{normalized_shape}, eps={eps}, ' \
#             'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
 