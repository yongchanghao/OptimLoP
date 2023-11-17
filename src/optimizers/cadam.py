import torch
from torch.optim.optimizer import Optimizer


class CAdam(Optimizer):
    r"""Implements CAdam algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99, 0.999), eps=1e-8, weight_decay=0.0):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99, 0.999))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = 0

                beta0, beta1, beta2 = group["betas"]
                eps = group["eps"]

                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # original adam:
                # updates = state["exp_avg"] / (1 - beta1 ** state["step"])
                # denom = state["exp_avg_sq"] / (1 - beta2 ** state["step"])
                # denom.sqrt_().add_(eps)
                # p.addcdiv_(updates, denom, value=-group["lr"])

                # our version:
                updates = state["exp_avg"] / (1 - beta1 ** state["step"])
                updates.mul_(beta0).add_(grad, alpha=(1 - beta0))

                denom = state["exp_avg_sq"] / (1 - beta2 ** state["step"])
                denom.mul_(beta0).add_(grad * grad, alpha=(1 - beta0))
                denom.sqrt_().add_(eps)

                p.addcdiv_(updates, denom, value=-group["lr"])

        return loss
