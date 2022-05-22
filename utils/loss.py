import torch

def weighted_class_bceloss(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        # loss = weights[1] * (target * torch.log(output)) + \
        #        weights[0] * ((1 - target) * torch.log(1 - output))
        loss = weights[1] * target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
               weights[0] * (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))

    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))
