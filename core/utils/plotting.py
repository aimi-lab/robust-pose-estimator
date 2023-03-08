import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
from skimage.transform import warp
from core.geometry.pinhole_transforms import create_img_coords_t, reproject, project


def _get_warpfield(depth: torch.Tensor, T: torch.Tensor, intrinsics: torch.Tensor, img_coords: torch.Tensor):
    # transform and project using pinhole camera model
    opts = reproject(depth, intrinsics, img_coords)
    return project(opts[:, :3], intrinsics, T)[:, :2]


def warp_frame(src_frame, depth, T, intrinsics):
    img_coordinates = create_img_coords_t(y=depth.shape[-2], x=depth.shape[-1])
    warpfield = _get_warpfield(depth.unsqueeze(0).detach().cpu(), T.unsqueeze(0).detach().cpu(), intrinsics[0].detach().cpu(), img_coordinates.detach().cpu()).squeeze().reshape(2, depth.shape[-2], depth.shape[-1])

    u = warpfield.numpy()[0].squeeze()
    v = warpfield.numpy()[1].squeeze()
    # get frame2 approximation by warping frame1 with the previous optical field
    src_img_warped = src_frame.float().detach().cpu().numpy()
    for ch in range(src_img_warped.shape[0]):
        src_img_warped[ch] = warp(src_img_warped[ch], np.array([v, u]), mode='edge')
    return torch.tensor(src_img_warped).to(torch.uint8)


def warp_frame_flow(src_frame, flow):
    h, w = flow.shape[-2:]
    row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    flow_off = torch.empty_like(flow)
    flow_off[1] = 2 * (flow[1] + row_coords.to(flow.device)) / (h - 1) - 1
    flow_off[0] = 2 * (flow[0] + col_coords.to(flow.device)) / (w - 1) - 1
    return torch.nn.functional.grid_sample(src_frame.unsqueeze(0).float(), flow_off.permute(1, 2, 0).unsqueeze(0), padding_mode='border', mode='nearest').squeeze().to(torch.uint8)


def plot_res(img1_batch,img2_batch, flow_batch, depth2_batch, pose_batch, conf1_batch, conf2_batch, intrinsics, n=2):

    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        return fig, axs
    flow_imgs = flow_to_image(flow_batch)
    img1_batch = [img.to(torch.uint8) for img in img1_batch[:n]]
    img2_batch = [img.to(torch.uint8) for img in img2_batch[:n]]
    conf1_batch = [(255 * img).to(torch.uint8) for img in conf1_batch[:n]]
    conf2_batch = [(255 * img).to(torch.uint8) for img in conf2_batch[:n]]
    img1_w_flow_batch = [warp_frame_flow(img, flow) for img, flow in zip(img1_batch[:2], flow_batch)]
    img1_w_pose_batch = [warp_frame(img, depth, pose, intrinsics) for img, depth, pose in zip(img1_batch, depth2_batch, pose_batch)]
    grid = [[img1, img2, img_w, img_w2, flow_img, conf1, conf2] for (img1, img2, img_w, img_w2, flow_img, conf1, conf2) in zip(img1_batch, img2_batch, img1_w_flow_batch, img1_w_pose_batch, flow_imgs[:n], conf1_batch, conf2_batch)]
    return plot(grid)
