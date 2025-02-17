import torch

def mse_score(output, target):
    # divide len
    with torch.no_grad():
        return torch.nn.functional.mse_loss(output, target).item()