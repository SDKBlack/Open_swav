import torch
from torch.optim.optimizer import Optimizer, required

class LARC(Optimizer):
    """
    LARC implementation
    """
    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        if trust_coefficient < 0.0:
            raise ValueError("Invalid trust_coefficient: {}".format(trust_coefficient))
        
        defaults = dict(trust_coefficient=trust_coefficient, clip=clip, eps=eps)
        self.optimizer = optimizer
        
        # Initialize super class to setup internal hooks
        super(LARC, self).__init__(optimizer.param_groups, defaults)

        self.param_groups = optimizer.param_groups
        self.defaults = defaults
        self.state = optimizer.state

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        with torch.no_grad():
            weight_decays = []
            for group in self.optimizer.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.defaults['trust_coefficient'] * (param_norm) / (grad_norm + param_norm * weight_decay + self.defaults['eps'])

                        # clip learning rate
                        if self.defaults['clip']:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optimizer.step()
        
        # restore weight decay to avoid messing up the optimizer
        for group, weight_decay in zip(self.optimizer.param_groups, weight_decays):
            group['weight_decay'] = weight_decay
