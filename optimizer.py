from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["t"] = 0.0
                    self.state[p]["m"] = torch.zeros_like(p.data, dtype=torch.float64)
                    self.state[p]["v"] = torch.zeros_like(p.data, dtype=torch.float64)
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                # Update first and second moments of the gradients
                state["t"] += 1.0
                m, v = state["m"], state["v"]
                t = state["t"]
                
                state["m"] = beta1*m + (1.0-beta1)*grad
                state["v"] = beta2*v + (1.0-beta2)*(grad**2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                
                a_t = alpha * math.sqrt(1.0-beta2**t) / (1.0-beta1**t)

                # Update parameters
                p.data -= a_t * m / (torch.sqrt(v) + eps)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                
                if group["weight_decay"] != 0:
                    p.data -= alpha * group["weight_decay"] * p.data

        return loss