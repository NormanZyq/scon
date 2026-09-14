import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, softmax
from .f_xy import f_xy


def gate_f(o_u, o_v, *, eps: float = 0.0, mode: str = "mono"):
    """
    Edge gate w in [0,1] from two endpoint scores o_u, o_v (assumed in [0,1]).

    mode:
      - "mono": low-low emphasis only (your f_xy). high-high and mixed -> small
      - "cos":  bimodal gate (low-low and high-high -> large). NOTE: not your original requirement
    """
    if mode == "mono":
        w = f_xy(o_u, o_v)
    elif mode == "cos":
        cu = torch.cos(torch.pi * o_u)
        cv = torch.cos(torch.pi * o_v)
        w = 0.5 * (cu * cv + 1.0)
    else:
        raise ValueError(f"Unknown gate mode: {mode}")
    if eps > 0:
        w = eps + (1.0 - eps) * w  # keep differentiable and ensures w>=eps

    return w


class _GatedWeightedMeanConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        gate_mode: str = "mono",
        eps: float = 0.0,
    ):
        super().__init__(aggr="add")  # sum aggregation
        self.lin = nn.Linear(in_channels, out_channels, bias=True)
        self.dropout = float(dropout)
        self.gate_mode = gate_mode
        self.eps = float(eps)

    def forward(self, x, edge_index, o):
        """
        x: [N, Fin]
        edge_index: [2, E]
        o: [N] in [0,1]  (node-wise gating score)
        """
        num_nodes = x.size(0)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index  # row=src, col=dst
        w = gate_f(o[row], o[col], mode=self.gate_mode, eps=self.eps)  # [E]

        if self.training and self.dropout > 0:
            keep = (torch.rand_like(w) >= self.dropout).to(dtype=w.dtype)
            w = w * keep

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, w=w, size=(num_nodes, num_nodes))

        # Weighted mean: divide by sum of weights per destination node
        denom = torch.zeros(num_nodes, device=w.device, dtype=w.dtype)
        denom.index_add_(0, col, w)  # denom[i] = sum_{j->i} w_{j->i}
        out = out / denom.clamp_min(1e-12).view(-1, 1)
        return out

    def message(self, x_j, w):
        """
        x_j: [E, F] source node features per edge
        w:   [E]    edge gate weights
        """
        return x_j * w.view(-1, 1)
    

class GatedGATConv(MessagePassing):
    """Multi-head GAT with confidence-style gating (no renormalization after gating)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        gate_mode: str = "mono",
        gate_eps: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.negative_slope = float(negative_slope)
        self.dropout = float(dropout)
        self.gate_mode = gate_mode
        self.gate_eps = float(gate_eps)

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_l = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.empty(1, heads, out_channels))

        if bias:
            bias_size = heads * out_channels if concat else out_channels
            self.bias = nn.Parameter(torch.empty(bias_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, o):
        num_nodes = x.size(0)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)

        alpha_l = (x * self.att_l).sum(dim=-1)
        alpha_r = (x * self.att_r).sum(dim=-1)

        out = self.propagate(
            edge_index,
            x=x,
            alpha=(alpha_l, alpha_r),
            o=o,
            size=(num_nodes, num_nodes),
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, alpha_i, alpha_j, o_i, o_j, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        g = gate_f(o_j, o_i, eps=self.gate_eps, mode=self.gate_mode)
        alpha = alpha * g.view(-1, 1)
        return x_j * alpha.unsqueeze(-1)


class GatedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        *,
        dropout: float,
        edge_dropout: float,
        gate_mode: str | bool,
        use_layernorm: bool,
        residual: bool,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual = bool(residual)
        self.dropout = float(dropout)
        self.use_layernorm = bool(use_layernorm)

        self.convs.append(
            _GatedWeightedMeanConv(in_channels, hidden_channels, dropout=edge_dropout, gate_mode=gate_mode)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                _GatedWeightedMeanConv(hidden_channels, hidden_channels, dropout=edge_dropout, gate_mode=gate_mode)
            )

        if self.use_layernorm:
            for _ in range(num_layers):
                self.norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, edge_index, o):
        h = x
        for layer_idx, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, o)
            if self.use_layernorm:
                h_new = self.norms[layer_idx](h_new)

            if layer_idx != len(self.convs) - 1:
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if self.residual and h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
        return h


class GatedGATEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        *,
        dropout: float,
        edge_dropout: float,
        gate_mode: str | bool,
        use_layernorm: bool,
        residual: bool,
        heads: int = 1,
        concat: bool = True,
        gate_eps: float = 0.0,
        negative_slope: float = 0.2,
        proj_after_concat: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual = bool(residual)
        self.dropout = float(dropout)
        self.use_layernorm = bool(use_layernorm)
        self.concat = bool(concat)
        self.heads = int(heads)
        self.proj_after_concat = bool(proj_after_concat)

        hidden_dim = hidden_channels * self.heads if self.concat else hidden_channels
        use_out_proj = self.proj_after_concat and hidden_dim != hidden_channels

        for layer_idx in range(num_layers):
            layer_in = in_channels if layer_idx == 0 else hidden_dim
            self.convs.append(
                GatedGATConv(
                    layer_in,
                    hidden_channels,
                    heads=self.heads,
                    concat=self.concat,
                    negative_slope=negative_slope,
                    dropout=edge_dropout,
                    gate_mode=gate_mode,
                    gate_eps=gate_eps,
                )
            )

        if self.use_layernorm:
            for _ in range(num_layers):
                self.norms.append(nn.LayerNorm(hidden_dim))

        if use_out_proj:
            self.out_proj = nn.Linear(hidden_dim, hidden_channels, bias=True)
        else:
            self.register_module("out_proj", None)

    def forward(self, x, edge_index, o):
        h = x
        for layer_idx, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, o)
            if self.use_layernorm:
                h_new = self.norms[layer_idx](h_new)

            if layer_idx != len(self.convs) - 1:
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if self.residual and h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
        if self.out_proj is not None:
            h = self.out_proj(h)
        return h
