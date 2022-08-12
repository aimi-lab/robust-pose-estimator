import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
from skimage.transform import warp
from alley_oop.photometry.raft.losses import _warp_frame, create_img_coords_t
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass


def warp_frame(src_frame, depth, T, intrinsics):
    img_coordinates = create_img_coords_t(y=depth.shape[-2], x=depth.shape[-1])
    warpfield = _warp_frame(depth.unsqueeze(0).detach().cpu(), T.unsqueeze(0).detach().cpu(), intrinsics.detach().cpu(), img_coordinates.detach().cpu()).squeeze().reshape(2, depth.shape[-2], depth.shape[-1])

    u = warpfield.numpy()[0].squeeze()
    v = warpfield.numpy()[1].squeeze()
    # get frame2 approximation by warping frame1 with the previous optical field
    src_img_warped = src_frame.float().detach().cpu().numpy()
    for ch in range(src_img_warped.shape[0]):
        src_img_warped[ch] = warp(src_img_warped[ch], np.array([v, u]), mode='edge')
    return torch.tensor(src_img_warped).to(torch.uint8)


def warp_frame_flow(src_frame, flow):
    u = flow.detach().cpu().numpy()[0].squeeze()
    v = flow.detach().cpu().numpy()[1].squeeze()
    nr, nc = src_frame.shape[-2:]
    import numpy as np
    from skimage.transform import warp
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    # get frame2 approximation by warping frame1 with the previous optical field
    src_img_warped = src_frame.float().detach().cpu().numpy()
    for ch in range(src_img_warped.shape[0]):
        src_img_warped[ch] = warp(src_img_warped[ch], np.array([row_coords + v, col_coords + u]), mode='edge')
    return torch.tensor(src_img_warped).to(torch.uint8)


def plot_res(img1_batch,img2_batch, flow_batch, depth2_batch, pose_batch, intrinsics, n=2):

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
    img1_w_flow_batch = [warp_frame_flow(img[:, ::8,::8], flow)for img, flow in zip(img1_batch[:2], flow_batch)]
    img1_w_pose_batch = [warp_frame(img[:, ::8,::8], depth, pose, intrinsics) for img, depth, pose in zip(img1_batch, depth2_batch, pose_batch)]
    grid = [[img1, img2, img_w, img_w2, flow_img] for (img1, img2, img_w, img_w2, flow_img) in zip(img1_batch, img2_batch, img1_w_flow_batch, img1_w_pose_batch, flow_imgs[:n])]
    return plot(grid)


def plot_3d(img1_batch,img2_batch, depth1_batch, depth2_batch, pose_batch, intrinsics, n=0):
    from viewer.viewer3d import Viewer3D
    viewer = Viewer3D((500,500), blocking=True)
    img1_pcl = SurfelMap(frame=FrameClass(img1_batch[None,n][..., ::8,::8]/255.0, depth1_batch[None,n], intrinsics=intrinsics), kmat=intrinsics).transform_cpy(pose_batch[n]).pcl2open3d(stable=False)
    img2_pcl = SurfelMap(frame=FrameClass(img2_batch[None,n][..., ::8,::8]/255.0, depth2_batch[None,n], intrinsics=intrinsics), kmat=intrinsics).pcl2open3d(stable=False)
    dists = np.asarray(img1_pcl.compute_point_cloud_distance(img2_pcl))
    print("mean pcl distance: ", dists.mean())
    viewer(pose=torch.eye(4), pcd=img1_pcl, add_pcd=img2_pcl)