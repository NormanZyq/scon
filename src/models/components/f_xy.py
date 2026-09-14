import torch


def smooth_gate_01(u: torch.Tensor, *, t: float = 0.5, k: float = 8.0) -> torch.Tensor:
    """
    Smooth gate g(u) in [0,1] for u in [0,1], with exact endpoint normalization:
      g(0)=1, g(1)=0, g is C-infinity.

    Args:
        u: torch.Tensor, assumed (typically) in [0,1]
        t: threshold in (0,1)
        k: steepness (>0). Smaller => smoother transition.

    Returns:
        g(u): torch.Tensor with values in [0,1] (for u in [0,1])
    """
    if not (0.0 < t < 1.0):
        raise ValueError(f"t must be in (0,1), got {t}")
    if k <= 0.0:
        raise ValueError(f"k must be > 0, got {k}")

    # Use dtype/device consistent scalars
    dtype, device = u.dtype, u.device
    t_t = torch.tensor(t, dtype=dtype, device=device)
    k_t = torch.tensor(k, dtype=dtype, device=device)

    # sigma(z) = 1/(1+exp(-z))
    sigma = torch.sigmoid

    # a = sigma(k*(t-0)) = sigma(k*t)
    # b = sigma(k*(t-1))
    a = sigma(k_t * (t_t - torch.tensor(0.0, dtype=dtype, device=device)))
    b = sigma(k_t * (t_t - torch.tensor(1.0, dtype=dtype, device=device)))

    denom = (a - b)
    # Extremely unlikely unless k is tiny and t near endpoints; still, guard numerical issues.
    if torch.isclose(denom, torch.zeros_like(denom)):
        raise ValueError("Numerical issue: a-b is too close to 0. Try larger k or different t.")

    return (sigma(k_t * (t_t - u)) - b) / denom


def f_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    t: float = 0.5,
    k: float = 8.0,
) -> torch.Tensor:
    """
    Monotonicity gate f(x,y) in [0,1] for x,y in [0,1].
    f(x,y) = g(x) * g(y), where g is a smooth gate in [0,1] on [0,1].

    Behavior:
      - x,y both small (<~t) -> f near 1
      - one large, one small -> f near 0
      - both large (>~t) -> f near 0

    Args:
        x, y: torch.Tensor (broadcastable). Typically in [0,1].
        t: threshold in (0,1)
        k: steepness (>0). Smaller => smoother transition.

    Returns:
        torch.Tensor: f(x,y) in [0,1] (for x,y in [0,1])
    """
    gx = smooth_gate_01(x, t=t, k=k)
    gy = smooth_gate_01(y, t=t, k=k)
    return gx * gy


# --- quick usage example ---
if __name__ == "__main__":
    x = torch.linspace(0, 1, 5)
    y = torch.linspace(0, 1, 5)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = f_xy(X, Y)  # defaults t=0.5, k=8.0
    print(Z)
