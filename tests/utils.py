from torch import optim

def spectral_GD(W, lr=0.001):
    """Spectral step gradient descent as seen in class"""
    lr_scale = max(1, W.shape[0] / W.shape[1]) ** 0.5
    return optim.Muon(
        [W], lr=lr / lr_scale,
        momentum=0.0, nesterov=False, weight_decay=0.0,
        ns_coefficients=(1.5, -0.5, 0.0), ns_steps=10,
        adjust_lr_fn="original",
    )

def sort_parameters(params):
    """Split params into two lists: the first contains the parameters that can
    be updated with spectral gradient descent (2D matrices), the second contains
    the rest
    """
    spectral_params = []
    other_params = []
    for p in params:
        if p.dim() == 2 and p.shape[0] > 1 and p.shape[1] > 1:
            spectral_params.append(p)
        else:
            other_params.append(p)
    return spectral_params, other_params
