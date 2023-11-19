import torch
from torch.optim.optimizer import Optimizer


class CSGD(Optimizer):
    r"""Implements CSGD algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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

                beta0, beta1 = group["betas"]

                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)

                # original sgd:
                # updates = state["exp_avg"]
                # p.add_(updates, alpha=-group["lr"])

                # our version:
                updates = state["exp_avg"]
                updates.mul_(beta0).add_(grad, alpha=1 - beta0)
                p.add_(updates, alpha=-group["lr"])

        return loss
