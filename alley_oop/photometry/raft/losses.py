import torch
from alley_oop.geometry.lie_3d import lie_se3_to_SE3_batch
from alley_oop.geometry.pinhole_transforms import create_img_coords_t


def _warp_frame(depth: torch.Tensor, T: torch.Tensor, intrinsics: torch.Tensor, img_coords: torch.Tensor):
    # transform and project using pinhole camera model
    # pinhole projection
    opts = _reproject(depth, intrinsics, img_coords)
    # compose projection matrix
    pmat = intrinsics[None,...] @ T[:,:3]

    # pinhole projection
    ipts = torch.bmm(pmat, opts)

    # inhomogenization
    ipts = ipts[:, :3] / ipts[:,-1].unsqueeze(1)
    return ipts[:,:2]


def _reproject(depth: torch.Tensor, intrinsics: torch.Tensor, img_coords: torch.Tensor):
    # transform and project using pinhole camera model
    # pinhole projection
    n = depth.shape[0]
    repr = torch.linalg.inv(intrinsics) @ img_coords.view(3, -1)
    opts = depth.view(n, 1, -1) * repr.unsqueeze(0)

    opts = torch.cat((opts, torch.ones((n, 1, opts.shape[-1]), device=opts.device, dtype=opts.dtype)), dim=1)
    return opts


def of_sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def supervised_pose_loss(pose_pred, pose_gt):
    return (pose_pred - pose_gt).abs().mean()


def l1_loss(pred, gt):
    return (pred - gt).abs().mean()


def seq_loss(loss_func, args, gamma=0.8):
    loss = 0.0
    n_predictions = len(args[0])
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        arguments = []
        for a in args:
            if isinstance(a, list):
                arguments.append(a[i])
            else:
                arguments.append(a)
        i_loss = loss_func(*arguments)
        loss += i_weight * (i_loss)
    return loss


def geometric_2d_loss(flow_preds, se3_preds, intrinsics, trg_depth, trg_confidence, valid):
    n = flow_preds.shape[0]
    img_coordinates = create_img_coords_t(y=trg_depth.shape[-2], x=trg_depth.shape[-1]).to(flow_preds.device).double()
    T_est = lie_se3_to_SE3_batch(se3_preds.double()/ 1000.0)  # invert transform to be consistent with other pose estimators
    warped_pts = _warp_frame(trg_depth.double(), T_est, intrinsics.double(),img_coordinates)
    residuals = torch.linalg.norm(img_coordinates[None,:2] + flow_preds.view(n,2, -1).double() - warped_pts, ord=2, dim=1)
    mask = torch.isnan(flow_preds[:, 0]).view(n, -1) | torch.isnan(flow_preds[:, 1]).view(n,-1) | ~valid.view(n, -1)
    # weight residuals by confidences
    residuals = torch.sqrt(trg_confidence.view(n, -1).double()) * residuals
    residuals[mask] = 0.0
    loss = torch.mean(residuals).float()
    return loss


def geometric_3d_loss(flow_preds, se3_preds, intrinsics, trg_depth, src_depth, trg_confidence, src_confidence, valid):
    n,_,h,w = flow_preds.shape
    # reproject to 3D
    T_est = lie_se3_to_SE3_batch(se3_preds.double()/ 1000.0)  #
    img_coordinates = create_img_coords_t(y=trg_depth.shape[-2], x=trg_depth.shape[-1]).to(flow_preds.device)
    trg_opts = _reproject(trg_depth, intrinsics, img_coordinates)
    trg_opts = torch.bmm(T_est, trg_opts.double()) #SurfelMap(frame=trg_frame, kmat=intrinsics, ignore_mask=True, pmat=T_est)
    ref_opts = _reproject(src_depth.double(), intrinsics.double(), img_coordinates.double())

    # get optical flow correspondences
    offset = torch.arange(h*w).reshape(1,1,h,w).to(flow_preds.device)
    flow_off = (flow_preds[:,1]*w).unsqueeze(1)
    flow_unravel = (offset+flow_off + flow_preds[:,0].unsqueeze(1)).view(n,1,-1).round().long().repeat(1,4,1)
    flow_unravel = flow_unravel.clamp(0,h*w-1)

    # compute residuals
    residuals = torch.linalg.norm(torch.gather(trg_opts, 2,flow_unravel)-ref_opts, ord=2, dim=1)
    mask = torch.isnan(flow_preds[:, 0]).view(n, -1) | torch.isnan(flow_preds[:, 1]).view(n, -1) | ~valid.view(n, -1)
    # weight residuals by confidences
    residuals = torch.sqrt(trg_confidence.view(n, -1)*torch.gather(src_confidence.view(n, -1), 1,flow_unravel[:,0])).double() * residuals
    residuals[mask] = 0.0
    loss = torch.mean(residuals).float()
    return loss