import torch
from torch.nn.functional import max_pool2d
from typing import Union, Tuple

from alley_oop.geometry.pinhole_transforms import forward_project2image, reverse_project, create_img_coords_t, forward_project
from alley_oop.interpol.sparse_img_interpolation import SparseImgInterpolator
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.utils.pytorch import batched_dot_product
from alley_oop.pose.frame_class import FrameClass


class SurfelMap(object):
    def __init__(self, *args, **kwargs):
        """ 
        https://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf
        http://thomaswhelan.ie/Whelan16ijrr.pdf
        """
        super().__init__()

        # consider input arguments
        self.opts = kwargs['opts'] if 'opts' in kwargs else torch.Tensor()
        dept = kwargs['dept'] if 'dept' in kwargs else torch.Tensor()
        self.device = self.opts.device if self.opts.numel() > 0 else dept.device
        dtype = self.opts.dtype if self.opts.numel() > 0 else dept.dtype
        mask = kwargs['mask'] if 'mask' in kwargs else torch.ones_like(dept).to(torch.bool)
        self.gray = kwargs['gray'] if 'gray' in kwargs else torch.Tensor().to(self.device)
        self.pmat = kwargs['pmat'] if 'pmat' in kwargs else torch.eye(4, dtype=dtype, device=self.device)  # extrinsics
        self.kmat = kwargs['kmat'] if 'kmat' in kwargs else torch.eye(3, dtype=dtype, device=self.device)     # intrinsics
        self.radi = kwargs['radi'] if 'radi' in kwargs else torch.Tensor().to(self.device)
        self.nrml = kwargs['normals'] if 'normals' in kwargs else torch.Tensor().to(self.device)
        self.conf_thr = kwargs['conf_thr'] if 'conf_thr' in kwargs else 10
        self.t_max = kwargs['t_max'] if 't_max' in kwargs else 15
        self.img_shape = kwargs['img_shape'] if 'img_shape' in kwargs else None
        self.upscale = kwargs['upscale'] if 'upscale' in kwargs else 4
        self.dbug_opt = False
        self.interpolate = SparseImgInterpolator(5, 2, 0)


        # calculate object points
        if dept.numel() > 0 and self.img_shape is not None:
            ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1]).to(self.device).to(dtype)
            self.opts = reverse_project(ipts=ipts, kmat=self.kmat, rmat=self.pmat[:3,:3],
                                        tvec=self.pmat[:3,3][..., None],
                                        dpth=dept.reshape(self.img_shape))[:, mask.view(-1)]
        elif dept.numel() == 0 and self.opts.numel() > 0 and self.radi.numel() == 0 and self.img_shape is not None:
            # rotate, translate and forward-project points
            dept = self.render().depth

        # initiliaze focal length
        self.flen = (self.kmat[0, 0] + self.kmat[1, 1]) / 2

        # initialize radii
        if self.radi.numel() == 0:
            if dept.numel() > 0 and self.nrml.numel() == 3*dept.numel():
                self.radi = ((dept.view(-1)) / (self.flen* 2**.5 * abs(self.nrml[2,:]))).unsqueeze(0)[:, mask.view(-1)]
            elif self.opts.numel() > 0:
                self.radi = torch.ones((1, self.opts.shape[1])).to(self.device)

        # initialize confidence
        self.conf = torch.Tensor()
        if self.opts.numel() > 0:
            gamma = self.opts[2].flatten()/torch.max(self.opts[2])
            self.conf = torch.exp(-.5 * gamma**2 / .6**2)[None ,:]

        # intialize tick as timestamp
        self.tick = 0
        if self.opts.numel() > 0:
            self.t_created = torch.zeros(1,self.opts.shape[1]).to(self.device)

    def fuse(self, dept: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor, pmat: torch.Tensor, mask: torch.Tensor=None):
        
        # prepare parameters
        self.img_shape = gray.shape[-2:] if self.img_shape is None else self.img_shape
        pmat_inv = torch.linalg.inv(pmat)
        kmat = self.kmat.clone()
        mask = torch.ones_like(dept).to(torch.bool) if mask is None else mask

        if self.upscale > 1:
            # consider upsampling
            gray = torch.nn.functional.interpolate(gray, scale_factor=self.upscale, mode='bilinear', align_corners=None)
            dept = torch.nn.functional.interpolate(dept, scale_factor=self.upscale, mode='bilinear', align_corners=None)
            mask = torch.nn.functional.interpolate(mask.float(), scale_factor=self.upscale, mode='nearest', align_corners=None).to(torch.bool)
            kmat[:2] *= self.upscale

        # prepare image and object coordinates
        ipts = create_img_coords_t(y=self.img_shape[-2]*self.upscale, x=self.img_shape[-1]*self.upscale).to(dept.dtype).to(dept.device)

        # project depth to 3d-points in world coordinates
        opts = reverse_project(ipts=ipts, dpth=dept, rmat=pmat[:3, :3], tvec=pmat[:3, -1][..., None], kmat=kmat)

        # update normals (if necessary)
        if normals is None or self.upscale > 1:
            normals = normals_from_regular_grid(opts.T.reshape((self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale, 3)), pad_opt=True).T

        # consider masked surfels and enforce channel x samples shape
        opts = opts[:, mask.view(-1)]
        gray = gray.flatten()[None, mask.view(-1)]
        normals = normals.reshape(3, -1)[:, mask.view(-1)]

        # rotate image normals to world-coordinates
        normals = pmat[:3, :3] @ normals

        # project all surfels to current image frame
        global_ipts = forward_project(self.opts, kmat=kmat, rmat=pmat_inv[:3, :3], tvec=pmat_inv[:3, -1][:, None])
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[1]*self.upscale-1) & (global_ipts[1, :] < self.img_shape[0]*self.upscale-1)

        # get correspondence by assigning projected points to image coordinates
        midx = self.get_match_indices(global_ipts[:, bidx])

        # compute mask that rejects depth and normal outliers
        vidx, midx = self.filter_surfels_by_correspondence(opts=opts, vidx=bidx, midx=midx, normals=normals)

        # compute radii
        radi = (opts[2, :] * 2**.5) / (self.flen * abs(normals[2, :]))[None, :]

        # compute confidence
        gamma = opts[2, :]/torch.max(opts[2, :])
        conf = torch.exp(-.5 * gamma**2 / .6**2)[None, :]

        # pre-select confidence elements
        ccor = conf[:, midx]
        conf_idx = self.conf[:, vidx]

        # update existing points, intensities, normals, radii and confidences
        self.opts[:, vidx] = (conf_idx*self.opts[:, vidx] + ccor*opts[:, midx]) / (conf_idx + ccor)
        self.gray[:, vidx] = (conf_idx*self.gray[:, vidx] + ccor*gray[:, midx]) / (conf_idx + ccor)
        self.radi[:, vidx] = (conf_idx*self.radi[:, vidx] + ccor*radi[:, midx]) / (conf_idx + ccor)
        self.nrml[:, vidx] = (conf_idx*self.nrml[:, vidx] + ccor*normals[:, midx]) / (conf_idx + ccor)
        self.conf[:, vidx] = conf_idx + ccor

        # create mask identifying unmatched indices
        mask = torch.zeros(opts.shape[1]).to(opts.device)
        mask[midx] = 1.0
        # bring mask back to original shape
        mask = ~max_pool2d(mask.view(1,1,self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale), self.upscale, stride=self.upscale).view(-1).to(torch.bool)

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

        ok_pts = ((self.conf > self.conf_thr) | ((self.tick - self.t_created) < self.t_max)).squeeze()
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
        d_thresh: float = 3,
        n_thresh: float = 30,
        remove_duplicates: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        filter correspondences by depth and normal angle
        """

        # parameter init
        vidx = torch.ones(self.opts.shape[1], dtype=bool) if vidx is None else vidx
        normals = torch.ones_like(self.opts) if normals is None else normals
        angle_threshold = torch.cos(torch.tensor(n_thresh)/180*torch.pi)

        # 1. depth distance constraint
        valid = torch.abs(opts[2, midx] - self.opts[2, vidx]) < d_thresh

        # 2. normals constraint (degrees threshold)
        valid &= torch.abs(batched_dot_product(normals[:, midx].T, self.nrml[:, vidx].T)) > angle_threshold

        # combine constraints and update indices
        vidx[vidx.clone()] &= valid
        midx = midx[valid]

        if remove_duplicates:
            # identify duplicates
            oidx, inv_idx, bins = torch.unique(midx, sorted=False, return_counts=True, return_inverse=True)
            duplicates = oidx[bins>1]
            candidate_mask = torch.ones_like(midx).to(torch.bool)
            kidx = vidx.clone()
            # TODO vectorize for-loop , this is too slow, I had to comment it out for the moment
            for d in duplicates:
                candidates = midx == d
                candidates &= ~(self.conf[:,vidx] == torch.max(self.conf[:,vidx][:,candidates])).squeeze()
                kidx[vidx] &= ~candidates
                candidate_mask &= ~candidates
            midx = midx[candidate_mask]
        else:
            kidx = vidx
        return kidx, midx

    @property
    def normals(self):
        return self.nrml

    @normals.setter
    def normals(self, normals):
        self.nrml = normals

    ########################

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
                         radi=self.radi).to(self.device)

    @property
    def grid_pts(self):
        assert self.img_shape is not None
        return self.opts.T.view((*self.img_shape, 3))

    @property
    def grid_normals(self):
        assert self.img_shape is not None
        return self.nrml.T.view((*self.img_shape, 3))

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
        return FrameClass(colors, depth, intrinsics=intrinsics, mask=mask).to(intrinsics.device)

    def pcl2open3d(self, stable: bool=True):
        """
        convert SurfelMap points to open3D PointCloud object for visualization
        :param stable: if True return only stable surfels
        """
        import open3d
        pcd = open3d.geometry.PointCloud()
        if stable:
            stable_pts = (self.conf > self.conf_thr).squeeze()
        else:
            stable_pts = torch.ones_like(self.conf, dtype=torch.bool).squeeze()
        pcd.points = open3d.utility.Vector3dVector(self.opts.T[stable_pts].cpu().numpy())
        if self.gray.numel() > 0:
            rgb = self.gray.repeat((3,1)).T[stable_pts]
            pcd.colors = open3d.utility.Vector3dVector(rgb.cpu().numpy())
        return pcd

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
