import math

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

def compute_struct_error(
    z,
    pos_edge_index,
    *,
    edge_index_for_neg=None,    # it is fine with edge_index itself
    num_nodes: int,
    num_pos_samples: int,
    neg_ratio: float,
):
    """
    Approximate per-node structural reconstruction error via link prediction loss.

    - Sample a subset of positive edges each call.
    - Sample negative edges with the same count * neg_ratio.
    - Use dot-product logits: s(u,v) = <z_u, z_v>/sqrt(d).
    - Return node-level error by averaging incident edge losses.
    """
    device = z.device
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be > 0")

    if edge_index_for_neg is None:
        edge_index_for_neg = pos_edge_index

    e = int(pos_edge_index.size(1))
    if e == 0:
        return torch.zeros(num_nodes, device=device, dtype=z.dtype), {"pos_edges": 0, "neg_edges": 0}

    if num_pos_samples <= 0 or num_pos_samples >= e:
        sampled_pos = pos_edge_index
    else:
        idx = torch.randint(0, e, (int(num_pos_samples),), device=device)
        sampled_pos = pos_edge_index[:, idx]

    u_pos, v_pos = sampled_pos
    mask = u_pos != v_pos
    if bool(mask.any()):
        u_pos = u_pos[mask]
        v_pos = v_pos[mask]

    num_pos = int(u_pos.numel())
    num_neg = int(round(num_pos * float(neg_ratio)))
    if num_pos == 0 or num_neg == 0:
        return torch.zeros(num_nodes, device=device, dtype=z.dtype), {"pos_edges": num_pos, "neg_edges": num_neg}

    neg_edge_index = negative_sampling(edge_index=edge_index_for_neg, num_nodes=num_nodes, num_neg_samples=num_neg)
    u_neg, v_neg = neg_edge_index

    scale = math.sqrt(max(int(z.size(-1)), 1))
    score_pos = (z[u_pos] * z[v_pos]).sum(dim=1) / scale
    score_neg = (z[u_neg] * z[v_neg]).sum(dim=1) / scale

    loss_pos = F.softplus(-score_pos)
    loss_neg = F.softplus(score_neg)

    err = torch.zeros(num_nodes, device=device, dtype=loss_pos.dtype)
    cnt = torch.zeros(num_nodes, device=device, dtype=loss_pos.dtype)

    ones_pos = torch.ones_like(loss_pos)
    err.index_add_(0, u_pos, loss_pos)
    err.index_add_(0, v_pos, loss_pos)
    cnt.index_add_(0, u_pos, ones_pos)
    cnt.index_add_(0, v_pos, ones_pos)

    ones_neg = torch.ones_like(loss_neg)
    err.index_add_(0, u_neg, loss_neg)
    err.index_add_(0, v_neg, loss_neg)
    cnt.index_add_(0, u_neg, ones_neg)
    cnt.index_add_(0, v_neg, ones_neg)

    struct_err = err / cnt.clamp_min(1.0)
    metrics = {
        "pos_edges": int(num_pos),
        "neg_edges": int(num_neg),
        "lp_loss_pos": float(loss_pos.mean().detach().cpu().item()),
        "lp_loss_neg": float(loss_neg.mean().detach().cpu().item()),
    }
    return struct_err, metrics


def recon_error(x, x_hat, s, s_hat,
                      pos_weight_a=0.5,
                      pos_weight_s=0.5,
                      bce_s=False,
                      use_lp_loss=False, **kwargs):
    # attribute reconstruction loss
    diff_attr = torch.pow(x - x_hat, 2)

    if pos_weight_a != 0.5:
        diff_attr = torch.where(x > 0, 
                                diff_attr * pos_weight_a, 
                                diff_attr * (1 - pos_weight_a))

    attr_error = torch.sqrt(torch.sum(diff_attr, 1))
    if use_lp_loss:
        pos_edge_index = kwargs.get('pos_edge_index')
        z = kwargs.get('z')
        num_nodes = x.shape[0]
        num_pos_samples = kwargs.get('num_pos_samples')
        neg_ratio = kwargs.get('neg_ratio')
        str_error = compute_struct_error(z, pos_edge_index, num_nodes=num_nodes, num_pos_samples=num_pos_samples, neg_ratio=neg_ratio)
    else:
        # structure reconstruction loss
        if bce_s:
            diff_stru = F.binary_cross_entropy(s_hat, s, reduction='none')
        else:
            diff_stru = torch.pow(s - s_hat, 2)

        if pos_weight_s != 0.5:
            diff_stru = torch.where(s > 0, 
                                    diff_stru * pos_weight_s, 
                                    diff_stru * (1 - pos_weight_s))

        str_error = torch.sqrt(torch.sum(diff_stru, 1))  
    return attr_error, str_error