import torch
from typing import Union

from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project, create_img_coords_t
from alley_oop.interpol.img_mappings import img_map_torch
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
        self.gray = kwargs['gray'] if 'gray' in kwargs else torch.Tensor().to(self.device)
        self.pmat = kwargs['pmat'] if 'pmat' in kwargs else torch.eye(4, dtype=dtype, device=self.device)  # extrinsics
        self.kmat = kwargs['kmat'] if 'kmat' in kwargs else torch.eye(3, dtype=dtype, device=self.device)     # intrinsics
        self.radi = kwargs['radi'] if 'radi' in kwargs else torch.Tensor().to(self.device)
        self.nrml = kwargs['normals'] if 'normals' in kwargs else torch.Tensor().to(self.device)
        self.img_shape = kwargs['img_shape'] if 'img_shape' in kwargs else None
        self.upscale = kwargs['upscale'] if 'upscale' in kwargs else 4
        self.dbug_opt = False


        # calculate object points
        if dept.numel() > 0 and self.img_shape is not None:
            ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1]).to(self.device).to(dtype)
            self.opts = reverse_project(ipts=ipts, kmat=self.kmat, rmat=torch.eye(3, dtype=dtype).to(self.device),
                                        tvec=torch.zeros((3, 1), dtype=dtype).to(self.device),
                                        dpth=dept.reshape(self.img_shape))
        elif dept.numel() == 0 and self.opts.numel() > 0 and self.radi.numel() == 0 and self.img_shape is not None:
            # rotate, translate and forward-project points
            dept = self.render().depth

        # initiliaze focal length
        self.flen = (self.kmat[0, 0] + self.kmat[1, 1]) / 2

        # initialize radii
        if self.radi.numel() == 0:
            if dept.numel() > 0 and self.nrml.numel() == 3*dept.numel():
                self.radi = ((dept.view(-1)) / (self.flen* 2**.5 * abs(self.nrml[2,:]))).unsqueeze(0)
            elif self.opts.numel() > 0:
                self.radi = torch.ones((1, self.opts.shape[1])).to(self.device)

        # initialize confidence
        self.conf = torch.Tensor()
        if self.opts.numel() > 0:
            gamma = self.opts[2].flatten()/torch.max(self.opts[2])
            self.conf = torch.exp(-.5 * gamma**2 / .6**2)[None ,:]

        # intialize tick as timestamp
        self.tick = 0

    def fuse(self, dept: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor = None, pmat: torch.Tensor = None):
        
        # update image shape
        self.img_shape = gray.shape[-2:] if self.img_shape is None else self.img_shape

        if self.upscale > 1:
            # consider upsampling
            gray = torch.nn.functional.interpolate(gray, scale_factor=self.upscale, mode='bilinear', align_corners=None)
            dept = torch.nn.functional.interpolate(dept, scale_factor=self.upscale, mode='bilinear', align_corners=None)

        # prepare image and object coordinates
        ipts = create_img_coords_t(y=self.img_shape[-2]*self.upscale, x=self.img_shape[-1]*self.upscale)
        opts = reverse_project(ipts=ipts, dpth=dept, rmat=self.pmat[:3, :3], tvec=self.pmat[:3, -1][..., None], kmat=self.kmat)

        # update normals (if necessary)
        if normals is None or self.upscale > 1:
            normals = normals_from_regular_grid(opts.reshape((self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale, 3)), pad_opt=True).T

        # enforce channel x samples shape
        gray = gray.flatten()[None, :]
        dept = dept.flatten()[None, :]
        normals = normals.reshape(3, -1)

        # project all surfels to current image frame
        global_ipts = forward_project(self.opts, kmat=self.kmat, rmat=pmat[:3, :3], tvec=pmat[:3, -1][:, None])
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[1]-1) & (global_ipts[1, :] < self.img_shape[0]-1)

        # find correspondence by projecting surfels to current frame
        midx = self.get_match_indices(global_ipts[:, bidx])

        # compute that rejects correspondences for a single unique one
        kidx = self.get_unique_correspondence_mask(opts=opts, vidx=bidx, midx=midx)
        # TODO: find strategy to combine bidx and kidx to single mask

        # re-compute matching indices while considering uniquness constraint
        midx = self.get_match_indices(global_ipts[:, bidx][:, kidx])

        # compute radii
        radi = (opts[2, :] * 2**.5) / (self.flen * abs(normals[2, :]))[None, :]

        # compute confidence
        gamma = opts[2, :]/torch.max(opts[2, :])
        conf = torch.exp(-.5 * gamma**2 / .6**2)[None, :]

        # select corresponding elements
        ocor, ncor, gcor, rcor, ccor = opts[:, midx], normals[:, midx], gray[:, midx], radi[:, midx], conf[:, midx]

        # update existing points, intensities, normals, radii and confidences
        conf_idx = self.conf[:, bidx][:, kidx]
        self.opts[:, bidx][:, kidx] = conf_idx*self.opts[:, bidx][:, kidx] + ccor*ocor / (conf_idx + ccor)
        self.gray[:, bidx][:, kidx] = conf_idx*self.gray[:, bidx][:, kidx] + ccor*gcor / (conf_idx + ccor)
        self.radi[:, bidx][:, kidx] = conf_idx*self.radi[:, bidx][:, kidx] + ccor*rcor / (conf_idx + ccor)
        self.nrml[:, bidx][:, kidx] = conf_idx*self.nrml[:, bidx][:, kidx] + ccor*ncor / (conf_idx + ccor)
        self.conf[:, bidx][:, kidx] = conf_idx + ccor

        # create mask identifying unmatched indices
        mask = torch.ones(opts.shape[1], dtype=bool)
        mask[midx.unique()] = False

        if self.dbug_opt:
            # print ratio of added points vs frame resolution
            ratio = mask[0::self.upscale**2].sum()/len(mask[0::self.upscale**2])
            print(ratio)

        # concatenate unmatched points, intensities, normals, radii and confidences
        self.opts = torch.cat((self.opts, opts[:, mask][0::self.upscale**2]), dim=-1)
        self.gray = torch.cat((self.gray, gray[:, mask][0::self.upscale**2]), dim=-1)
        self.radi = torch.cat((self.radi, radi[:, mask][0::self.upscale**2]), dim=-1)
        self.nrml = torch.cat((self.nrml, normals[:, mask][0::self.upscale**2]), dim=-1)
        self.conf = torch.cat((self.conf, conf[:, mask][0::self.upscale**2]), dim=-1)

        self.tick = self.tick + 1

    def get_match_indices(
        self,
        ipts: torch.Tensor = None,
        ) -> torch.Tensor:

        # quantize points (while considering super-sampling factor)
        ipts_quantized = torch.round((ipts-.5)*self.upscale)

        # get point correspondence from indexing as flattened 2D indices
        midx = ipts_quantized[1, :] * self.img_shape[1] * self.upscale + ipts_quantized[0, :]

        return midx.long()
    
    def filter_corresponding_points(
        self,
        opts: torch.Tensor = None,
        normals: torch.Tensor = None,
        midx: torch.Tensor = None,
        vidx: torch.Tensor = None,
        d_thresh: float = 1, 
        n_thresh: float = 20,
        ) -> torch.Tensor:

        # 1. depth distance constraint
        didx = abs(opts[2] - self.opts[2, vidx][midx]) < d_thresh

        # 2. normals constraint (20 degrees threshold)
        nidx = batched_dot_product(normals.T[midx], self.nrml.T[vidx]) > torch.cos(n_thresh/180*torch.pi)

        # combine constraints
        fidx = vidx[vidx.clone()] & didx & nidx

        return fidx

    def get_unique_correspondence_mask(
        self,
        opts: torch.Tensor,
        midx: torch.Tensor,
        vidx: torch.Tensor = None,
        normals: torch.Tensor = None,
        d_thresh: float = 1,
        n_thresh: float = 20,
        ) -> torch.Tensor:
        """
        yields mask for a unique correspondence assignment to exclude points mapping to the same 2-D pixel location
        """

        # parameter init
        vidx = torch.ones(self.opts.shape[1], dtype=bool) if vidx is None else vidx
        normals = torch.ones(self.opts.shape)
        angle_threshold = torch.cos(torch.tensor(n_thresh)/180*torch.pi)

        # identify duplicates 
        oidx, bins = torch.unique(midx, sorted=False, return_counts=True)
        duplicates = oidx[bins>1]

        kidx = torch.ones(self.opts[:, vidx].shape[1], dtype=bool)
        # TODO vectorize for-loop
        for d in duplicates:
            # 1. depth distance constraint
            candidates = abs(opts[2, d] - self.opts[2, vidx][midx == d]) < d_thresh
            # 2. normals constraint (20 degrees threshold)
            if torch.sum(candidates == torch.min(candidates)) > 1:
                candidates = batched_dot_product(normals[:, d][:, None].T, self.nrml[:, vidx][:, midx==d].T) > angle_threshold
            # 3. confidence constraint
            if torch.sum(candidates == torch.min(candidates)) > 1:
                candidates = self.conf[:, vidx][:, midx == d]
            # 4. euclidean distance constraint (if necessary)
            if torch.sum(candidates == torch.min(candidates)) > 1:
                candidates = torch.sum((opts[:, d][:, None] - self.opts[:, vidx][:, midx == d])**2, dim=0)**.5
            # pick first element if there is no unique candidate
            if torch.sum(candidates == torch.min(candidates)) > 1:
                candidates = torch.arange(len(candidates))
            mask = torch.ones(self.opts[:, vidx].shape[1], dtype=bool)
            mask[torch.where(midx==d)[0][torch.argmin(candidates.long())]] = False
            kidx[(midx == d) & mask] = 0

        return kidx

    @property
    def normals(self):
        return self.nrml

    @normals.setter
    def normals(self, normals):
        self.nrml = normals

    ########################

    def transform(self, transform):
        assert transform.shape == (4,4)
        self.opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        self.nrml = transform[:3,:3] @ self.nrml

    def transform_cpy(self, transform):
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
        ####
        if intrinsics is None:
            intrinsics = self.kmat
        if extrinsics is None:
            extrinsics = self.pmat

        # rotate, translate and forward-project points
        pts_h = torch.vstack([self.opts, torch.ones(self.opts.shape[1], dtype=self.opts.dtype, device=self.device)])
        npts = forward_project(pts_h, kmat=intrinsics, rmat=extrinsics[:3, :3],
                               tvec=extrinsics[:3, -1][..., None], inhomogenize_opt=True)
        depth = img_map_torch(img=self.opts[2].view((1, 1, *self.img_shape)), npts=npts, mode='bilinear').view((1, 1, *self.img_shape))
        colors = img_map_torch(img=self.gray.view((1, 1, *self.img_shape)), npts=npts, mode='bilinear').view((1, 1, *self.img_shape))

        return FrameClass(colors, depth, intrinsics=intrinsics).to(intrinsics.device)

    def pcl2open3d(self):
        import open3d
        pcd = open3d.geometry.PointCloud()
        #pcd.normals = open3d.utility.Vector3dVector(self.nrml.cpu().numpy())
        pcd.points = open3d.utility.Vector3dVector(self.opts.T.cpu().numpy())
        rgb = self.gray.unsqueeze(1).repeat((1,3))
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
        return self
