import torch


def batched_dot_product(t1:torch.tensor, t2:torch.tensor):
    return (t1.unsqueeze(1) @ t2.unsqueeze(-1)).squeeze()
