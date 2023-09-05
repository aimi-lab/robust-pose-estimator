import torch
from torch.nn.functional import max_pool2d
from lietorch import SE3
from typing import Union, Tuple
import numpy as np
import warnings

from core.geometry.pinhole_transforms import project2image, reproject, create_img_coords_t, project, transform
from core.interpol.sparse_img_interpolation import SparseImgInterpolator
from core.utils.frame_class import Frame
from core.utils.save_ply import save_ply


class SurfelMap(object):
    def __init__(self, *args, **kwargs):
        """ 
        https://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf
        http://thomaswhelan.ie/Whelan16ijrr.pdf
        """
        super().__init__()
        self.pmat = kwargs['pmat'] if 'pmat' in kwargs else SE3.Identity(1)  # extrinsics
        self.conf_thr = kwargs['conf_thr'] if 'conf_thr' in kwargs else 7
        self.t_max = kwargs['t_max'] if 't_max' in kwargs else 15
        self.upscale = kwargs['upscale'] if 'upscale' in kwargs else 1
        self.d_thresh = kwargs['d_thresh'] if 'd_thresh' in kwargs else 100.0
        self.dbug_opt = False
        self.interpolate = SparseImgInterpolator(5, 2, 0)
        self.depth_scale = kwargs['depth_scale'] if 'depth_scale' in kwargs else 1.0  # only used for open3d pcl
        self.patch_colors = None
        self.average_points = kwargs['average_pts'] if 'average_pts' in kwargs else True
        # either provide opts, normals and color or a frame class
        if 'opts' in kwargs:
            assert 'rgb' in kwargs
            self.kmat = kwargs['kmat']
            self.opts = kwargs['opts']
            self.device = self.opts.device
            dtype = self.opts.dtype
            self.rgb = kwargs['rgb']
            self.img_shape = kwargs['img_shape'] if 'img_shape' in kwargs else None
            # initialize confidence
            if 'conf' in kwargs:
                self.conf = kwargs['conf']
            else:
                gamma = self.opts[2].flatten() / torch.max(self.opts[2])
                self.conf = torch.exp(-.5 * gamma ** 2 / .6 ** 2)[None, :]
        else:
            assert 'frame' in kwargs
            assert 'kmat' in kwargs
            self.kmat = kwargs['kmat']
            frame = kwargs['frame']
            self.device = frame.depth.device
            dtype = frame.depth.dtype
            self.img_shape = frame.shape
            # check ignore mask option, this is important if points are required in 2D grid shape
            ignore_mask = kwargs['ignore_mask'] if 'ignore_mask' in kwargs else False
            mask = torch.ones_like(frame.depth).to(torch.bool) if ignore_mask else frame.mask
            # calculate object points
            ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1]).to(self.device).to(dtype)

            opts = reproject(img_coords=ipts, intrinsics=self.kmat.to(self.device).to(dtype), depth=frame.depth)
            self.opts = transform(opts, self.pmat.to(self.device).to(dtype)[None,...]).squeeze()[:3, mask.view(-1)]

            self.rgb = frame.img.view(3, -1)[:, mask.view(-1)]
            self.conf = frame.confidence[mask].view(1, -1) / self.conf_thr  # normalize confidence

        self.kmat = self.kmat.to(self.device).to(dtype)
        self.pmat = self.pmat.to(self.device).to(dtype)

        # intialize tick as timestamp
        self.tick = 0
        self.t_created = torch.zeros(1,self.opts.shape[1]).to(self.device)

    def fuse(self, *args):
        frame, pose = args[:2]

        # prepare parameters
        self.img_shape = frame.shape
        pose_inv = pose.inv()
        kmat = self.kmat.clone()

        rgb = frame.img
        depth = frame.depth
        mask = frame.mask
        dtype = depth.dtype

        if self.upscale > 1:
            # consider upsampling
            rgb = torch.nn.functional.interpolate(rgb, scale_factor=self.upscale, mode='bilinear', align_corners=None)
            depth = torch.nn.functional.interpolate(depth, scale_factor=self.upscale, mode='bilinear', align_corners=None)
            kmat[:2] *= self.upscale

        # prepare image and object coordinates
        ipts = create_img_coords_t(y=self.img_shape[-2]*self.upscale, x=self.img_shape[-1]*self.upscale).to(dtype).to(self.device)

        # project depth to 3d-points in world coordinates
        opts = reproject(img_coords=ipts, intrinsics=self.kmat, depth=depth)
        opts = transform(opts, pose[None,...]).squeeze()[:3]

        # consider masked surfels and enforce channel x samples shape
        rgb = rgb.view(3, -1)

        # project all surfels to current image frame
        global_ipts = project(self.opts.unsqueeze(0), intrinsics=kmat.unsqueeze(0), T=pose_inv[None,...]).squeeze()
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[1]*self.upscale-1) & (global_ipts[1, :] < self.img_shape[0]*self.upscale-1)

        # get correspondence by assigning projected points to image coordinates
        midx = self.get_match_indices(global_ipts[:, bidx])

        # compute mask that rejects depth and normal outliers and detects duplicated surface patches
        vidx, midx = self.filter_surfels_by_correspondence(opts=opts, vidx=bidx, midx=midx, d_thresh=self.d_thresh)

        # apply frame mask to reject invalid pixels
        bidx[vidx.clone()] &= (frame.mask.view(-1)[(midx/self.upscale**2).long()]).type(torch.bool)
        midx = midx[frame.mask.view(-1)[(midx/self.upscale**2).long()]]

        # pre-select confidence elements
        conf = torch.ones_like(depth.view(1, -1)) / self.conf_thr
        ccor = conf[:, midx]
        conf_idx = self.conf[:, vidx]

        # update existing points, intensities, normals, radii and confidences
        if self.average_points:
            self.opts[:, vidx] = (conf_idx*self.opts[:, vidx] + ccor*opts[:, midx]) / (conf_idx + ccor)
            self.rgb[:, vidx] = (conf_idx * self.rgb[:, vidx] + ccor * rgb[:, midx]) / (conf_idx + ccor)
        self.conf[:, vidx] = torch.clamp(conf_idx + ccor, 0.0, 1.0)  # saturate confidence to 1

        # create mask identifying unmatched indices
        mask = torch.zeros(opts.shape[1]).to(opts.device)
        mask[midx] = 1.0
        # bring mask back to original shape
        mask = ~max_pool2d(mask.view(1,1,self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale), self.upscale, stride=self.upscale).view(-1).to(torch.bool)
        # avoid fusion with invalid pixels
        mask &= frame.mask.view(-1)
        if self.dbug_opt:
            # print ratio of added points vs frame resolution
            ratio = mask.float().mean()
            print(ratio)

        # concatenate unmatched points, intensities, normals, radii and confidences
        self.opts = torch.cat((self.opts, self._downsample(opts)[:, mask]), dim=-1)
        self.rgb = torch.cat((self.rgb, self._downsample(rgb)[:, mask]), dim=-1)
        self.conf = torch.cat((self.conf, self._downsample(conf)[:, mask]), dim=-1)
        self.t_created = torch.cat((self.t_created, self.tick*torch.ones(1, mask.sum()).to(self.device)), dim=-1)

        self.tick = self.tick + 1

        # remove surfels
        self.remove_surfels_by_confidence_and_time()

    def remove_surfels_by_confidence_and_time(self):
        """ remove unstable points that have been created long time ago """

        ok_pts = ((self.conf >= 1.0) | ((self.tick - self.t_created) < self.t_max)).squeeze()
        self.opts = self.opts[:, ok_pts]
        self.rgb = self.rgb[:, ok_pts]
        self.conf = self.conf[:, ok_pts]
        self.t_created = self.t_created[:, ok_pts]
        return ok_pts

    def _downsample(self, x):
        x = x.view(-1, self.img_shape[0] * self.upscale, self.img_shape[1] * self.upscale)
        x = x[:, ::self.upscale, ::self.upscale].reshape(x.shape[0], -1)
        return x

    def get_match_indices(
        self,
        ipts: torch.Tensor = None,
        upscale: int = None
        ) -> torch.Tensor:

        if upscale is None:
            upscale = self.upscale
        # quantize points (while considering super-sampling factor)
        ipts_quantized = torch.round((ipts-.5))

        # get point correspondence from indexing as flattened 2D indices
        midx = ipts_quantized[1, :] * self.img_shape[1] * upscale + ipts_quantized[0, :]

        return midx.long()

    def filter_surfels_by_correspondence(
        self,
        opts: torch.Tensor,
        midx: torch.Tensor,
        vidx: torch.Tensor = None,
        d_thresh: float = 0.05,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        filter correspondences by depth and normal angle
        """

        # parameter init
        vidx = torch.ones(self.opts.shape[1], dtype=bool) if vidx is None else vidx

        # 1. depth distance constraint
        depth_diff = opts[2, midx] - self.opts[2, vidx]
        valid_depth = torch.abs(depth_diff) < d_thresh

        # combine constraints and update indices
        vidx[vidx.clone()] &= valid_depth
        midx = midx[valid_depth]

        return vidx, midx

    def transform(self, tr:SE3):
        """
        transform surfels (3d points and normals) inplace
        :param transform: lietorch SE3 transform
        """
        self.opts = transform(self.opts.unsqueeze(0), tr[None,...]).squeeze()

    def transform_cpy(self, tr:torch.tensor):
        """
        transform surfels (3d points and normals) and return a copy
        :param transform: lietorch SE3 transform
        """
        opts = transform(self.opts.unsqueeze(0), tr[None,...]).squeeze()
        return self._constructor(opts=opts, kmat=self.kmat, rgb=self.rgb, img_shape=self.img_shape,
                         depth_scale=self.depth_scale, conf=self.conf).to(self.device)

    @property
    def grid_pts(self):
        assert self.img_shape is not None
        return self.opts.T.view((*self.img_shape, 3))

    @property
    def confidence(self):
        return self.conf.view(-1)

    def render(self, intrinsics: torch.tensor=None, extrinsics: SE3=None):
        """
        render frame (image, depth and mask) from surfel-map
        :param intrinsics: camera intrinsics
        :param extrinsics: camera extrinsics
        """
        if intrinsics is None:
            intrinsics = self.kmat
        if extrinsics is None:
            extrinsics = self.pmat

        # rotate, translate and forward-project points
        sort_idx = torch.argsort(self.conf, dim=1)[0]
        pts_h = self.opts[:, sort_idx]
        npts, valid = project2image(pts_h.unsqueeze(0), img_shape=self.img_shape, intrinsics=intrinsics.unsqueeze(0), T=extrinsics[None,...])
        npts = npts.squeeze()
        valid = valid.squeeze()
        # generate sparse img maps and interpolate missing values
        img_coords = npts[1, valid].long(), npts[0, valid].long()
        confidence = torch.zeros(self.img_shape, dtype=self.opts.dtype, device=self.device)
        confidence[img_coords] = self.conf[0, sort_idx][valid]

        depth = torch.zeros(self.img_shape, dtype=self.opts.dtype, device=self.device)
        # confidence aware rendering.
        depth[img_coords] = self.opts[2, sort_idx][valid]
        mask = confidence[None,None,...] != 0.0
        depth = self.interpolate(depth[None,None,...])

        colors = torch.zeros((3, *self.img_shape), dtype=self.opts.dtype, device=self.device)
        colors[0][img_coords] = self.rgb[0, sort_idx][valid]
        colors[1][img_coords] = self.rgb[1, sort_idx][valid]
        colors[2][img_coords] = self.rgb[2, sort_idx][valid]
        colors = self.interpolate(colors[None,...])

        return Frame(colors, depth=depth, mask=mask, confidence=confidence[None, None, ...]).to(intrinsics.device), None

    def pcl2open3d(self, stable: bool=True, filter: torch.Tensor=None):
        """
        convert SurfelMap points to open3D PointCloud object for visualization
        :param stable: if True return only stable surfels
        """
        import open3d
        pcd = open3d.geometry.PointCloud()
        if filter is None:
            filter = torch.ones_like(self.conf, dtype=torch.bool).squeeze()
        if stable:
            stable_pts = (self.conf[:,filter] >= 1.0).squeeze()
        else:
            stable_pts = torch.ones_like(self.conf[:,filter], dtype=torch.bool).squeeze()

        pcd.points = open3d.utility.Vector3dVector(self.opts.T[filter][stable_pts].cpu().numpy()/self.depth_scale)
        if self.rgb.numel() > 0:
            rgb = self.rgb.T[filter][stable_pts]
            pcd.colors = open3d.utility.Vector3dVector(rgb.cpu().numpy()/255.0)
        return pcd

    def save_ply(self, path:str, stable:bool=True):
        if stable:
            stable_pts = (self.conf >= 1.0).squeeze()
        else:
            stable_pts = torch.ones_like(self.conf, dtype=torch.bool).squeeze()
        opts = self.opts.T[stable_pts].cpu().numpy()/self.depth_scale
        if self.rgb.numel() > 0:
            rgb = self.rgb.T[stable_pts].cpu().numpy()
        else:
            rgb = np.zeros_like(opts)
        if (len(opts) > 0) & (len(rgb) > 0):
            save_ply(opts, rgb, path)

    def to(self, d: Union[torch.device, torch.dtype]):
        self.rgb = self.rgb.to(d)
        self.kmat = self.kmat.to(d)
        self.opts = self.opts.to(d)
        self.pmat = self.pmat.to(d)
        self.device = self.opts.device
        self.interpolate = self.interpolate.to(d)
        self.conf = self.conf.to(d)
        self.t_created = self.t_created.to(d)
        return self

    @property
    def _constructor(self):
        return SurfelMap
