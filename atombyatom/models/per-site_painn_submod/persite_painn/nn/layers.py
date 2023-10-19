from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_


zeros_initializer = partial(constant_, val=0.0)
DEFAULT_DROPOUT_RATE = 0.0
EPS = 1e-15


class CosineEnvelope(nn.Module):
    # Behler, J. Chem. Phys. 134, 074106 (2011)
    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, d):

        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0

        return output


class Dense(nn.Linear):
    """Applies a dense layer with activation: :math:`y = activation(Wx + b)`
    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.
        Returns:
            torch.Tensor: Output of the dense layer.
        """
        self.to(inputs.device)
        y = super().forward(inputs)

        # kept for compatibility with earlier versions of nff
        if hasattr(self, "dropout"):
            y = self.dropout(y)

        if self.activation:
            y = self.activation(y)

        return y


class PainnRadialBasis(nn.Module):
    def __init__(self, n_rbf, cutoff, learnable_k):
        super().__init__()

        self.n = torch.arange(1, n_rbf + 1).float()
        if learnable_k:
            self.n = nn.Parameter(self.n)

        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        # replace divide by 0 with limit of sinc function

        denom = torch.where(shape_d == 0, torch.tensor(1.0, device=device), shape_d)
        num = torch.where(shape_d == 0, coef, torch.sin(coef * shape_d))

        output = torch.where(
            shape_d >= self.cutoff, torch.tensor(0.0, device=device), num / denom
        )

        return output


def to_module(activation):
    from persite_painn.nn.builder import LAYERS_TYPE

    return LAYERS_TYPE[activation]()
