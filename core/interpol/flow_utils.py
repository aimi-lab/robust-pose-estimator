import torch
from core.utils.pytorch import grid_sample

def remap_from_flow(x, flow):
    # get optical flow correspondences
    n, _, h, w = flow.shape
    row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    flow_off = torch.empty_like(flow)
    flow_off[:, 1] = 2 * (flow[:, 1] + row_coords.to(flow.device)) / (h - 1) - 1
    flow_off[:, 0] = 2 * (flow[:, 0] + col_coords.to(flow.device)) / (w - 1) - 1
    x = torch.nn.functional.grid_sample(x, flow_off.permute(0, 2, 3, 1), align_corners=True)
    #x = grid_sample(x, flow_off.permute(0, 2, 3, 1))
    valid = (x > 0).any(dim=1).unsqueeze(1)
    return x, valid


def remap_from_flow_nearest(x, flow):
    # get optical flow correspondences
    n, _, h, w = flow.shape
    row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    flow_off = torch.empty_like(flow)
    flow_off[:, 1] = 2 * (flow[:, 1] + row_coords.to(flow.device)) / (h - 1) - 1
    flow_off[:, 0] = 2 * (flow[:, 0] + col_coords.to(flow.device)) / (w - 1) - 1
    x = torch.nn.functional.grid_sample(x.float(), flow_off.permute(0, 2, 3, 1), align_corners=True, mode='nearest')
    valid = (x > 0).any(dim=1).unsqueeze(1)
    return x, valid