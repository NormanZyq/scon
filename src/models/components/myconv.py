import torch
from typing import Optional
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv

import torch.nn as nn
import torch.nn.functional as F


class TwoLayerGNN(nn.Module):
    """
    Two-layer configurable GNN (PyG).

    Args:
        in_channels: input feature dim
        hidden_channels: hidden dim (for GAT this is per-head dim)
        out_channels: output dim
        backbone: 'gcn' | 'gat' | 'sage' | 'graph'
        heads: number of heads for GAT (ignored for others)
        dropout: dropout after first layer
        use_bn: whether to apply BatchNorm after first layer
        act: activation function (callable)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        backbone: str = "gcn",
        heads: int = 1,
        dropout: float = 0.5,
        use_bn: bool = False,
        act=F.relu,
    ):
        super().__init__()
        backbone = backbone.lower()
        self.backbone = backbone
        self.act = act
        self.dropout = dropout
        self.use_bn = use_bn

        if backbone == "gat":
            # conv1 outputs hidden_channels * heads
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
            conv1_out = hidden_channels * heads
            # produce out_channels, disabling concat in final layer
            self.conv2 = GATConv(conv1_out, out_channels, heads=1, concat=False)
        elif backbone == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        self.bn = nn.BatchNorm1d(hidden_channels * heads) if (use_bn and backbone == "gat" and heads > 1) else \
                  nn.BatchNorm1d(hidden_channels) if use_bn else None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        return_embeddings: bool = False,
    ) -> Tensor:
        # conv1 (try to pass edge_weight when supported)
        try:
            x = self.conv1(x, edge_index, edge_weight)
        except TypeError:
            x = self.conv1(x, edge_index)

        if self.act is not None:
            x = self.act(x)

        if self.bn is not None:
            x = self.bn(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # conv2
        try:
            out = self.conv2(x, edge_index, edge_weight)
        except TypeError:
            out = self.conv2(x, edge_index)

        if return_embeddings:
            return out, x  # (logits, hidden_embeddings)
        return out


class MultiLayerGNN(nn.Module):
    """
    Multi-layer configurable GNN (PyG).

    Args:
        in_channels: input feature dim
        hidden_channels: hidden dim(s) for intermediate layers
        out_channels: output dim
        num_layers: total number of layers (including output)
        backbone: 'gcn' | 'gat' | 'sage' | 'graph'
        heads: number of heads for GAT (ignored for others)
        dropout: dropout after hidden layers
        use_bn: whether to apply BatchNorm after hidden layers
        act: activation function (callable)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | list[int] | tuple[int, ...],
        out_channels: int,
        num_layers: int = 2,
        backbone: str = "gcn",
        heads: int = 1,
        dropout: float = 0.5,
        use_bn: bool = False,
        act=F.relu,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        backbone = backbone.lower()
        self.backbone = backbone
        self.act = act
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.num_layers = int(num_layers)

        if isinstance(hidden_channels, (list, tuple)):
            hidden_dims = [int(h) for h in hidden_channels]
            expected = max(self.num_layers - 1, 0)
            if len(hidden_dims) != expected:
                raise ValueError("hidden_channels length must match num_layers - 1")
        else:
            hidden_dims = [int(hidden_channels)] * max(self.num_layers - 1, 0)

        def make_conv(in_ch: int, out_ch: int, is_last: bool) -> nn.Module:
            if backbone == "gat":
                if is_last:
                    return GATConv(in_ch, out_ch, heads=heads, concat=False)
                return GATConv(in_ch, out_ch, heads=heads)
            if backbone == "gcn":
                return GCNConv(in_ch, out_ch)
            if backbone == "sage":
                return SAGEConv(in_ch, out_ch)
            if backbone == "graph":
                return GraphConv(in_ch, out_ch)
            raise ValueError(f"Unknown backbone: {backbone}")

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if (self.use_bn and self.num_layers > 1) else None
        self._use_edge_weight = backbone in ("gcn", "graph")

        layer_in = int(in_channels)
        for layer_idx in range(self.num_layers):
            is_last = layer_idx == self.num_layers - 1
            layer_out = int(out_channels) if is_last else hidden_dims[layer_idx]
            self.convs.append(make_conv(layer_in, layer_out, is_last))

            if not is_last:
                if self.bns is not None:
                    bn_dim = layer_out * heads if backbone == "gat" else layer_out
                    self.bns.append(nn.BatchNorm1d(bn_dim))
                layer_in = layer_out * heads if backbone == "gat" else layer_out

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        return_embeddings: bool = False,
    ) -> Tensor:
        convs = self.convs
        bns = self.bns
        act = self.act
        use_edge_weight = self._use_edge_weight and edge_weight is not None
        dropout_p = self.dropout
        training = self.training

        h = x
        hidden = None
        for layer_idx, conv in enumerate(convs):
            is_last = layer_idx == len(convs) - 1
            if use_edge_weight:
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)

            if not is_last:
                if act is not None:
                    h = act(h)
                if bns is not None:
                    h = bns[layer_idx](h)
                h = F.dropout(h, p=dropout_p, training=training)
                if return_embeddings:
                    hidden = h

        if return_embeddings:
            return h, hidden
        return h
