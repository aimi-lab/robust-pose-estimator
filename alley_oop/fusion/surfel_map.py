import torch
from torch.nn.functional import max_pool2d
from typing import Union, Tuple
import numpy as np

from alley_oop.geometry.pinhole_transforms import forward_project2image, reverse_project, create_img_coords_t, forward_project
from alley_oop.interpol.sparse_img_interpolation import SparseImgInterpolator
from alley_oop.geometry.normals import normals_from_regular_grid, resize_normalmap
from alley_oop.utils.pytorch import batched_dot_product
from alley_oop.pose.frame_class import FrameClass
from alley_oop.utils.save_ply import save_ply


class SurfelMap(object):
    def __init__(self, *args, **kwargs):
        """ 
        https://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf
        http://thomaswhelan.ie/Whelan16ijrr.pdf
        """
        super().__init__()
        self.pmat = kwargs['pmat'] if 'pmat' in kwargs else torch.eye(4)  # extrinsics
        self.kmat = kwargs['kmat'] if 'kmat' in kwargs else torch.eye(3)  # intrinsics
        self.conf_thr = kwargs['conf_thr'] if 'conf_thr' in kwargs else 7
        self.t_max = kwargs['t_max'] if 't_max' in kwargs else 15
        self.upscale = kwargs['upscale'] if 'upscale' in kwargs else 4
        self.d_thresh = kwargs['d_thresh'] if 'd_thresh' in kwargs else 100.0
        self.dbug_opt = False
        self.interpolate = SparseImgInterpolator(5, 2, 0)
        # initiliaze focal length
        self.flen = (self.kmat[0, 0] + self.kmat[1, 1]) / 2
        self.depth_scale = kwargs['depth_scale'] if 'depth_scale' in kwargs else 1.0  # only used for open3d pcl
        self.patch_colors = None
        # either provide opts, normals and color or a frame class
        if 'opts' in kwargs:
            assert 'normals' in kwargs
            assert 'gray' in kwargs

            self.opts = kwargs['opts']
            self.device = self.opts.device
            dtype = self.opts.dtype
            self.gray = kwargs['gray']
            self.nrml = kwargs['normals']
            self.radi = kwargs['radi'] if 'radi' in kwargs else torch.ones((1, self.opts.shape[1])).to(self.device)
            self.img_shape = kwargs['img_shape'] if 'img_shape' in kwargs else None
            # initialize confidence
            if 'conf' in kwargs:
                self.conf = kwargs['conf']
            else:
                gamma = self.opts[2].flatten() / torch.max(self.opts[2])
                self.conf = torch.exp(-.5 * gamma ** 2 / .6 ** 2)[None, :]
        else:
            assert 'frame' in kwargs
            frame = kwargs['frame']
            self.device = frame.depth.device
            dtype = frame.depth.dtype
            self.img_shape = frame.shape
            # check ignore mask option, this is important if points are required in 2D grid shape
            ignore_mask = kwargs['ignore_mask'] if 'ignore_mask' in kwargs else False
            mask = torch.ones_like(frame.depth).to(torch.bool) if ignore_mask else frame.mask
            # calculate object points
            ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1]).to(self.device).to(dtype)
            self.opts = reverse_project(ipts=ipts, kmat=self.kmat.to(self.device).to(dtype), rmat=self.pmat[:3, :3].to(self.device).to(dtype),
                                        tvec=self.pmat[:3, 3][..., None].to(self.device).to(dtype),
                                        dpth=frame.depth.squeeze())[:, mask.view(-1)]

            self.gray = frame.img_gray[mask].view(1, -1)
            self.nrml = frame.normals.view(3, -1)[:, mask.view(-1)]
            self.conf = frame.confidence[mask].view(1, -1) / self.conf_thr  # normalize confidence
            # initialize radii
            self.radi = ((frame.depth[mask].view(-1)) / (self.flen* 2**.5 * abs(self.nrml[2,:]))).unsqueeze(0)

        self.kmat = self.kmat.to(self.device).to(dtype)
        self.pmat = self.pmat.to(self.device).to(dtype)

        # intialize tick as timestamp
        self.tick = 0
        self.t_created = torch.zeros(1,self.opts.shape[1]).to(self.device)

    def fuse(self, frame: FrameClass, pmat: torch.Tensor):
        assert pmat.shape == (4,4)

        # prepare parameters
        self.img_shape = frame.shape
        pmat_inv = torch.linalg.inv(pmat)
        kmat = self.kmat.clone()

        gray = frame.img_gray
        depth = frame.depth
        mask = frame.mask
        normals = frame.normals
        dtype = depth.dtype

        if self.upscale > 1:
            # consider upsampling
            gray = torch.nn.functional.interpolate(gray, scale_factor=self.upscale, mode='bilinear', align_corners=None)
            depth = torch.nn.functional.interpolate(depth, scale_factor=self.upscale, mode='bilinear', align_corners=None)
            mask = torch.nn.functional.interpolate(mask.float(), scale_factor=self.upscale, mode='nearest', align_corners=None).to(torch.bool)
            kmat[:2] *= self.upscale

        # prepare image and object coordinates
        ipts = create_img_coords_t(y=self.img_shape[-2]*self.upscale, x=self.img_shape[-1]*self.upscale).to(dtype).to(self.device)

        # project depth to 3d-points in world coordinates
        opts = reverse_project(ipts=ipts, dpth=depth, rmat=pmat[:3, :3], tvec=pmat[:3, -1][..., None], kmat=kmat)

        # update normals (if necessary)
        if normals is None or self.upscale > 1:
            normals = normals_from_regular_grid(opts.T.reshape((self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale, 3)), pad_opt=True).T

        # consider masked surfels and enforce channel x samples shape
        gray = gray.view(1, -1)
        normals = normals.reshape(3, -1)

        # rotate image normals to world-coordinates
        normals = pmat[:3, :3] @ normals

        # project all surfels to current image frame
        global_ipts = forward_project(self.opts, kmat=kmat, rmat=pmat_inv[:3, :3], tvec=pmat_inv[:3, -1][:, None])
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[1]*self.upscale-1) & (global_ipts[1, :] < self.img_shape[0]*self.upscale-1)

        # get correspondence by assigning projected points to image coordinates
        midx = self.get_match_indices(global_ipts[:, bidx])

        # compute mask that rejects depth and normal outliers and detects duplicated surface patches
        vidx, midx, dmask = self.filter_surfels_by_correspondence(opts=opts, vidx=bidx, midx=midx, normals=normals,
                                                                  d_thresh=self.d_thresh, check_duplicate_surfaces=True)

        # apply frame mask to reject invalid pixels
        bidx[vidx.clone()] &= (frame.mask.view(-1)[(midx/self.upscale**2).long()]).type(torch.bool)
        midx = midx[frame.mask.view(-1)[(midx/self.upscale**2).long()]]

        # compute radii
        radi = (opts[2, :] * 2**.5) / (self.flen * abs(normals[2, :]))[None, :]

        # pre-select confidence elements
        conf = frame.confidence.view(1, -1) / self.conf_thr
        ccor = conf[:, midx]
        conf_idx = self.conf[:, vidx]

        # update existing points, intensities, normals, radii and confidences
        self.opts[:, vidx] = (conf_idx*self.opts[:, vidx] + ccor*opts[:, midx]) / (conf_idx + ccor)
        self.gray[:, vidx] = (conf_idx*self.gray[:, vidx] + ccor*gray[:, midx]) / (conf_idx + ccor)
        self.radi[:, vidx] = (conf_idx*self.radi[:, vidx] + ccor*radi[:, midx]) / (conf_idx + ccor)
        self.nrml[:, vidx] = (conf_idx*self.nrml[:, vidx] + ccor*normals[:, midx]) / (conf_idx + ccor)
        self.conf[:, vidx] = torch.clamp(conf_idx + ccor, 0.0, 1.0)  # saturate confidence to 1

        # create mask identifying unmatched indices
        mask = torch.zeros(opts.shape[1]).to(opts.device)
        mask[midx] = 1.0
        # bring mask back to original shape
        mask = ~max_pool2d(mask.view(1,1,self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale), self.upscale, stride=self.upscale).view(-1).to(torch.bool)
        # avoid fusion with invalid pixels
        mask &= frame.mask.view(-1) & dmask
        if self.dbug_opt:
            # print ratio of added points vs frame resolution
            ratio = mask.float().mean()
            print(ratio)

        # concatenate unmatched points, intensities, normals, radii and confidences
        self.opts = torch.cat((self.opts, self._downsample(opts)[:, mask]), dim=-1)
        self.gray = torch.cat((self.gray, self._downsample(gray)[:, mask]), dim=-1)
        self.radi = torch.cat((self.radi, self._downsample(radi)[:, mask]), dim=-1)
        self.nrml = torch.cat((self.nrml, self._downsample(normals)[:, mask]), dim=-1)
        self.conf = torch.cat((self.conf, self._downsample(conf)[:, mask]), dim=-1)
        self.t_created = torch.cat((self.t_created, self.tick*torch.ones(1, mask.sum()).to(self.device)), dim=-1)

        self.tick = self.tick + 1

        # remove surfels
        self.remove_surfels_by_confidence_and_time()

    def remove_surfels_by_confidence_and_time(self):
        """ remove unstable points that have been created long time ago """

        ok_pts = ((self.conf >= 1.0) | ((self.tick - self.t_created) < self.t_max)).squeeze()
        self.opts = self.opts[:, ok_pts]
        self.gray = self.gray[:, ok_pts]
        self.radi = self.radi[:, ok_pts]
        self.nrml = self.nrml[:, ok_pts]
        self.conf = self.conf[:, ok_pts]
        self.t_created = self.t_created[:, ok_pts]

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
        normals: torch.Tensor = None,
        d_thresh: float = 0.05,
        n_thresh: float = 20,
        remove_duplicates: bool = False,
        check_duplicate_surfaces: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        filter correspondences by depth and normal angle
        """

        # parameter init
        vidx = torch.ones(self.opts.shape[1], dtype=bool) if vidx is None else vidx
        normals = torch.ones_like(self.opts) if normals is None else normals
        angle_threshold = torch.cos(torch.tensor(n_thresh)/180*torch.pi)

        # 1. depth distance constraint
        depth_diff = opts[2, midx] - self.opts[2, vidx]
        valid_depth = torch.abs(depth_diff) < d_thresh

        # 2. normals constraint (degrees threshold)
        valid_normals = torch.abs(batched_dot_product(normals[:, midx].T, self.nrml[:, vidx].T)) > angle_threshold

        # 3. check for duplicated surfaces
        if check_duplicate_surfaces:
            invidx = vidx.clone()
            invidx[vidx] &= ~valid_depth
            duplicate_mask = self.detect_duplicated_surfaces(normals, invidx, depth_diff, midx)
            print(duplicate_mask.float().mean())
            if duplicate_mask.float().mean() < 0.3:
                #ToDo how do we handle failed slam registration?
                print(duplicate_mask.float().mean(), "  something went wrong here!!!")
        else:
            duplicate_mask = torch.ones(self.img_shape[0]*self.img_shape[1], device=self.device, dtype=torch.bool)

        # combine constraints and update indices
        vidx[vidx.clone()] &= valid_depth & valid_normals
        midx = midx[valid_depth & valid_normals]

        if remove_duplicates:
            # identify duplicates
            oidx, inv_idx, bins = torch.unique(midx, sorted=False, return_counts=True, return_inverse=True)
            duplicates = oidx[bins>1]
            candidate_mask = torch.ones_like(midx).to(torch.bool)
            kidx = vidx.clone()
            # TODO vectorize for-loop , this is too slow
            for d in duplicates:
                candidates = midx == d
                candidates &= ~(self.conf[:,vidx] == torch.max(self.conf[:,vidx][:,candidates])).squeeze()
                kidx[vidx] &= ~candidates
                candidate_mask &= ~candidates
            midx = midx[candidate_mask]
        else:
            kidx = vidx
        return kidx, midx, duplicate_mask

    @property
    def normals(self):
        return self.nrml

    @normals.setter
    def normals(self, normals):
        self.nrml = normals

    def detect_duplicated_surfaces(
            self,
            normals: torch.tensor,
            invidx: torch.tensor,
            depth_diff: torch.tensor,
            midx: torch.tensor,
            angle_threshold:float=45.0):
        """
            Detect duplicated surfaces and exclude them from surfel creation.
        """
        angle_threshold = torch.cos(torch.tensor(angle_threshold) / 180 * torch.pi)

        # 2. check if invalid surfels correspond to true occlusion by checking the normal directions

        # generate low-resolution normal maps from input frame and rendered scene
        patch_size = (self.img_shape[0]//32, self.img_shape[1]//32)
        normals_lowscale = resize_normalmap(normals.reshape(1,3,self.img_shape[0], self.img_shape[1]), patch_size)

        # rotate, translate and forward-project points
        pts_h = torch.vstack([self.opts[:, invidx], torch.ones(self.opts[:, invidx].shape[1], dtype=self.opts.dtype, device=self.device)])
        npts, idx = forward_project2image(pts_h, img_shape=self.img_shape, kmat=self.kmat)

        # generate sparse img maps and interpolate missing values
        img_coords = npts[1, idx].long() * self.img_shape[1] + npts[0, idx].long()
        scene_normals_low_scale = torch.zeros((3,*self.img_shape), device=self.device).view(3,-1)
        scene_normals_low_scale[:, img_coords] = self.normals[:, invidx][:, idx]

        scene_normals_low_scale = resize_normalmap(scene_normals_low_scale.view(1,3,*self.img_shape), patch_size)

        # check if normals are pointing in same direction
        duplicate_mask = torch.abs(batched_dot_product(normals_lowscale.view(3,-1).T, scene_normals_low_scale.view(3,-1).T)) < angle_threshold

        duplicate_mask = torch.nn.functional.interpolate(duplicate_mask.float().view(1,1,*patch_size), self.img_shape, mode='nearest')

        # 1. check if invalid surfels lie behind object such that they cannot be visible
        duplicate_mask = duplicate_mask.to(torch.bool).view(-1)
        duplicate_mask[midx] &= depth_diff < 0
        return duplicate_mask

    def transform(self, transform:torch.tensor):
        """
        transform surfels (3d points and normals) inplace
        :param transform: 4x4 homogenous transform
        """
        assert transform.shape == (4,4)
        self.opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        self.nrml = transform[:3,:3] @ self.nrml

    def transform_cpy(self, transform:torch.tensor):
        """
        transform surfels (3d points and normals) and return a copy
        :param transform: 4x4 homogenous transform
        """
        assert transform.shape == (4, 4)
        opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        normals = transform[:3,:3] @ self.nrml
        return SurfelMap(opts=opts, normals=normals, kmat=self.kmat, gray=self.gray, img_shape=self.img_shape,
                         radi=self.radi, depth_scale=self.depth_scale, conf=self.conf).to(self.device)

    @property
    def grid_pts(self):
        assert self.img_shape is not None
        return self.opts.T.view((*self.img_shape, 3))

    @property
    def grid_normals(self):
        assert self.img_shape is not None
        return self.nrml.T.view((*self.img_shape, 3))
    @property
    def confidence(self):
        return self.conf.view(-1)

    def render(self, intrinsics: torch.tensor=None, extrinsics: torch.tensor=None):
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
        pts_h = torch.vstack([self.opts, torch.ones(self.opts.shape[1], dtype=self.opts.dtype, device=self.device)])
        npts, valid = forward_project2image(pts_h, img_shape=self.img_shape, kmat=intrinsics, rmat=extrinsics[:3, :3],
                               tvec=extrinsics[:3, -1][..., None])

        # generate sparse img maps and interpolate missing values
        img_coords = npts[1, valid].long(), npts[0, valid].long()
        depth = torch.nan*torch.ones(self.img_shape, dtype=self.opts.dtype, device=self.device)
        depth[img_coords] = self.opts[2, valid]
        mask = ~torch.isnan(depth[None,None,...]).to(depth.device)
        depth = self.interpolate(depth[None,None,...])

        colors = torch.nan*torch.ones(self.img_shape, dtype=self.opts.dtype, device=self.device)
        colors[img_coords] = self.gray[0, valid]
        colors = self.interpolate(colors[None,None,...])

        confidence = torch.zeros(self.img_shape, dtype=self.opts.dtype, device=self.device)
        confidence[img_coords] = self.conf[0, valid]
        return FrameClass(colors, depth, intrinsics=intrinsics, mask=mask, confidence=confidence[None,None,...]).to(intrinsics.device)

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
            stable_pts = (self.conf[:,filter] > 1.0).squeeze()
        else:
            stable_pts = torch.ones_like(self.conf[:,filter], dtype=torch.bool).squeeze()

        pcd.points = open3d.utility.Vector3dVector(self.opts.T[filter][stable_pts].cpu().numpy()/self.depth_scale)
        if self.gray.numel() > 0:
            rgb = self.gray.repeat((3,1)).T[filter][stable_pts]
            pcd.colors = open3d.utility.Vector3dVector(rgb.cpu().numpy())
        return pcd

    def save_ply(self, path:str, stable:bool=True):
        if stable:
            stable_pts = (self.conf > 1.0).squeeze()
        else:
            stable_pts = torch.ones_like(self.conf, dtype=torch.bool).squeeze()
        opts = self.opts.T[stable_pts].cpu().numpy()/self.depth_scale
        if self.gray.numel() > 0:
            rgb = self.gray.repeat((3, 1)).T[stable_pts].cpu().numpy()
        else:
            rgb = np.zeros_like(opts)
        save_ply(opts, rgb, path)

    def to(self, d: Union[torch.device, torch.dtype]):
        self.radi = self.radi.to(d)
        self.nrml = self.nrml.to(d)
        self.gray = self.gray.to(d)
        self.kmat = self.kmat.to(d)
        self.opts = self.opts.to(d)
        self.pmat = self.pmat.to(d)
        self.device = self.opts.device
        self.interpolate = self.interpolate.to(d)
        self.conf = self.conf.to(d)
        self.t_created = self.t_created.to(d)
        return self
