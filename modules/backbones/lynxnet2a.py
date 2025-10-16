__COPYRIGHT__ = """
 _    Lynxnet-2a                     _    ____         
| | _   _  ____  __  __ ____    ___ | |_ |___ \   ____ 
| || | | ||  _ \ \ \/ /|  _ \  / _ \| __|  __) | / _  |
| || |_| || | | | |  | | | | ||  __/| |_  / __/ | (_| |
|_| \__  ||_| |_|/_/\_\|_| |_| \___| \__||_____| \____|
    |___/
            (C) Project VsingerXiaoice Group. SSPL-1.0 License.
"""
class RightPrinter:
    is_printed = False
    @staticmethod
    def printC():
        if not RightPrinter.is_printed:
            RightPrinter.is_printed = True
            print(__COPYRIGHT__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from modules.commons.common_layers import SinusoidalPosEmb, SwiGLU, ATanGLU, Transpose
from utils.hparams import hparams

class AmiyaConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.left = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.right = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.decide = nn.Sequential(
            nn.Conv1d(dim, 45, 1),
            nn.SiLU(),
            nn.Conv1d(45, dim, 1)
        )
    def forward(self, x):
        x = self.left(x)
        y = self.right(x)
        return x + (y - x) * torch.sigmoid(self.decide(x))

class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0., glu_type='swiglu'):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if glu_type == 'swiglu':
            _glu = SwiGLU()
        elif glu_type == 'atanglu':
            _glu = ATanGLU()
        else:
            raise ValueError(f'{glu_type} is not a valid activation')
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            Transpose((1, 2)),
            AmiyaConv(dim, kernel_size),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            _glu,
            nn.Linear(inner_dim, inner_dim * 2),
            _glu,
            nn.Linear(inner_dim, dim),
            _dropout
        )
        self.pgate = nn.Parameter(torch.rand(dim), requires_grad=True)
        self.gating = nn.Sequential(
            nn.Linear(dim, 45),
            nn.SiLU(),
            Transpose((1, 2)),
            nn.AdaptiveAvgPool1d(1),
            Transpose((1, 2)),
            nn.Linear(45, 90),
            _glu,
            nn.Linear(45, dim),
            nn.Sigmoid()
        )
    def forward(self, res):
        x = self.norm(res)
        y = self.net(x)
        return res + y * self.gating(y + x * self.pgate)


class LYNXNet2a(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, expansion_factor=1, kernel_size=31,
                 dropout=0.0, use_conditioner_cache=False, glu_type='swiglu', save_memory: bool = False):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.use_conditioner_cache = use_conditioner_cache
        if self.use_conditioner_cache:
            # It may need to be modified at some point to be compatible with the condition cache
            self.conditioner_projection = nn.Conv1d(hparams['hidden_size'], num_channels, 1)
        else:
            self.conditioner_projection = nn.Linear(hparams['hidden_size'], num_channels)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    glu_type=glu_type
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.zeros_(self.output_projection.weight)
        self.save_memory = save_memory
    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        RightPrinter.printC()

        if self.n_feats == 1:
            x = spec[:, 0]  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]

        x = self.input_projection(x.transpose(1, 2)) # [B, T, F x M]
        if self.use_conditioner_cache:
            # It may need to be modified at some point to be compatible with the condition cache
            x = x + self.conditioner_projection(cond).transpose(1, 2)
        else:
            x = x + self.conditioner_projection(cond.transpose(1, 2))
        x = x + self.diffusion_embedding(diffusion_step).unsqueeze(1)

        for layer in self.residual_layers:
            if self.save_memory and torch.is_grad_enabled() and not torch.onnx.is_in_onnx_export():
                x = cp.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
