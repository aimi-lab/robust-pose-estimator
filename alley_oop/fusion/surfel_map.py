import torch
from typing import Union

from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project, create_img_coords_t
from alley_oop.interpol.img_mappings import img_map_torch
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.utils.pytorch import batched_dot_product


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
        self.gray = kwargs['gray'] if 'gray' in kwargs else torch.Tensor().to(self.device)
        self.pmat = kwargs['pmat'] if 'pmat' in kwargs else torch.eye(4).to(self.device)    # extrinsics
        self.kmat = kwargs['kmat'] if 'kmat' in kwargs else torch.eye(3).to(self.device)      # intrinsics
        self.radi = kwargs['radi'] if 'radi' in kwargs else torch.Tensor().to(self.device)
        self.normals = kwargs['normals'] if 'normals' in kwargs else torch.Tensor().to(self.device)
        self.img_shape = kwargs['img_shape'] if 'img_shape' in kwargs else None
        self.upscale = kwargs['upscale'] if 'upscale' in kwargs else 1   # TODO: enable value other than 1
        self.dbug_opt = False


        # calculate object points
        if dept.numel() > 0 and self.img_shape is not None:
            ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1]).to(self.device)
            self.opts = reverse_project(ipts=ipts, kmat=self.kmat.float(), rmat=torch.eye(3).to(self.device),
                                        tvec=torch.zeros(3, 1).to(self.device)  ,
                                        dpth=dept.reshape(self.img_shape).float()).to(dept.dtype)
        elif dept.numel() == 0 and self.opts.numel() > 0 and self.radi.numel() == 0 and self.img_shape is not None:
            # rotate, translate and forward-project points
            npts = forward_project(self.opts.float(), kmat=self.kmat.float(), rmat=self.pmat[:3, :3],
                                   tvec=self.pmat[:3, -1][..., None], inhomogenize_opt=True).to(self.opts.dtype)
            dept = img_map_torch(img=npts[2].reshape((1,1,*self.img_shape)), npts=npts, mode='bilinear')

        # initiliaze focal length
        self.flen = (self.kmat[0, 0] + self.kmat[1, 1]) / 2

        # initialize radii
        if self.radi.numel() == 0:
            if dept.numel() > 0 and self.normals.numel() == 3*dept.numel():
                self.radi = (dept.view(-1)) / (self.flen* 2**.5 * abs(self.normals[2,:]))
            elif self.opts.numel() > 0:
                self.radi = torch.ones((1, self.opts.shape[1])).to(self.device)

        # initialize confidence
        self.conf = torch.Tensor()
        if self.opts.numel() > 0:
            gamma = self.opts[2].flatten()/torch.max(self.opts[2])
            self.conf = torch.exp(-.5 * gamma**2 / .6**2)[None ,:]

        # upsample value

        # intialize tick as timestamp
        self.tick = 0
            
    def fuse(self, dept: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor = None, pmat: torch.Tensor = None):
        
        # update image shape
        self.img_shape = gray.shape[-2:] if self.img_shape is None else self.img_shape

        # compute opts considering upsampling
        ipts = create_img_coords_t(y=self.img_shape[-2]*self.upscale, x=self.img_shape[-1]*self.upscale)
        ipts[:2, :] -= .5
        dept = torch.nn.functional.upsample(dept, scale_factor=self.upscale, mode='bilinear', align_corners=None)
        opts = reverse_project(ipts=ipts, dpth=dept, rmat=self.pmat[:3, :3], tvec=self.pmat[:3, -1][..., None], kmat=self.kmat)

        if normals is None:
            # TODO: ensure normals have same length as opts etc.
            normals = normals_from_regular_grid(opts.reshape((self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale, 3)))
        
        # enforce channel x samples shape
        normals = normals.reshape(3, -1)
        gray = gray.flatten()[None, :]
        dept = dept.flatten()[None, :]

        # project all surfels to current image frame
        global_ipts = forward_project(self.opts, kmat=self.kmat, rmat=pmat[:3, :3], tvec=pmat[:3, -1][:, None])
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[1]-1) & (global_ipts[1, :] < self.img_shape[0]-1)

        # find correspondence by projecting surfels to current frame
        midx = self.get_match_indices(global_ipts[:, bidx])                     # image border constraints
        fidx = bidx#self.filter_corresponding_points(opts=opts, normals=normals, midx=midx, vidx=bidx)
        kidx = self.get_unique_correspondence_mask(opts=opts, vidx=fidx, midx=midx)
        midx = self.get_match_indices(global_ipts[:, fidx][:, kidx])   # filter constraints

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
        self.normals[:, bidx][:, kidx] = conf_idx*self.normals[:, bidx][:, kidx] + ccor*ncor / (conf_idx + ccor)
        self.conf[:, bidx][:, kidx] = conf_idx + ccor

        # concatenate unmatched points, intensities, normals, radii and confidences
        mask = torch.ones(opts.shape[1], dtype=bool)
        mask[midx.unique()] = False

        if self.dbug_opt:
            # print ratio of added points vs resolution of current frame
            ratio = mask.sum()/len(mask)
            print(ratio)

        self.opts = torch.cat((self.opts, opts[:, mask]), dim=-1)
        self.gray = torch.cat((self.gray, gray[:, mask]), dim=-1)
        self.radi = torch.cat((self.radi, radi[:, mask]), dim=-1)
        self.normals = torch.cat((self.normals, normals[:, mask]), dim=-1)
        self.conf = torch.cat((self.conf, conf[:, mask]), dim=-1)

        self.tick = self.tick + 1

    def get_match_indices(
        self,
        ipts: torch.Tensor = None,
        ):

        # quantize points (while considering super-sampling factor)
        ipts_quantized = torch.round(ipts*self.upscale)

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
        ):

        # 1. depth distance constraint
        didx = abs(opts[2] - self.opts[2, vidx][midx]) < d_thresh

        # 2. normals constraint (20 degrees threshold)
        nidx = batched_dot_product(normals.T[midx], self.normals.T[vidx]) > torch.cos(n_thresh/180*torch.pi)

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
        ):
        """
        yields indices of points mapping to the same pixel location to enforce unique correspondence assignment 
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
                candidates = batched_dot_product(normals[:, d][:, None].T, self.normals[:, vidx][:, midx==d].T) > angle_threshold
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
            mask[torch.where(midx==d)[0][torch.argmin(torch.as_tensor(candidates, dtype=torch.int))]] = False
            kidx[(midx == d) & mask] = 0

        return kidx

    def _test_global_point_projection(self, global_ipts, vidx=None):

        vidx = torch.ones(global_ipts.shape[1])
        midx = self.get_match_indices(global_ipts[:, vidx])
        gpts = global_ipts[:, vidx][:, midx]
        timg = img_map_torch(img=gpts[2].reshape(self.img_shape)[None, None, ...], npts=gpts)
        import matplotlib.pyplot as plt
        plt.imshow(timg.cpu().numpy()[0 ,0 , ...])
        plt.show()

    ########################

    def transform(self, transform):
        assert transform.shape == (4,4)
        self.opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        self.normals = transform[:3,:3] @ self.normals

    def transform_cpy(self, transform):
        assert transform.shape == (4, 4)
        opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        normals = transform[:3,:3] @ self.normals
        return SurfelMap(opts=opts, normals=normals, kmat=self.kmat, gray=self.gray, img_shape=self.img_shape,
                         radi=self.radi).to(self.device)

    #def set_colors(self, colors):
    #    self.colors = torch.nn.Parameter(colors.squeeze().permute(1,2,0).reshape(-1, 3))

    @property
    def grid_pts(self):
        assert self.img_shape is not None
        return self.opts.T.view((*self.img_shape, 3))

    @property
    def grid_normals(self):
        assert self.img_shape is not None
        return self.normals.T.view((*self.img_shape, 3))

    def render(self, intrinsics):
        from alley_oop.geometry.pinhole_transforms import forward_project
        extrinsics = torch.eye(4).to(self.pts.dtype).to(self.pts.device)
        rmat = extrinsics[:3, :3]
        tvec = extrinsics[:3, 3, None]
        pts_h = torch.vstack([self.pts.T, torch.ones(self.pts.shape[0],
                                                           device=self.pts.device, dtype=self.pts.dtype)])
        points_2d = forward_project(pts_h, intrinsics, rmat=rmat, tvec=tvec).T

        # filter points that are not in the image
        valid = (points_2d[:, 1] < self.grid_shape[0]) & (points_2d[:, 0] < self.grid_shape[1]) & (
                    points_2d[:, 1] > 0) & (points_2d[:, 0] > 0)
        points_2d = points_2d[valid][...,:2]
        colors = self.colors[valid]

        import numpy as np
        x_coords = np.arange(0, self.grid_shape[1]) + 0.5
        y_coords = np.arange(0, self.grid_shape[0]) + 0.5
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
        interp = NDInterpolator(points_2d, colors, dist_thr=10, default_value=0)
        interp.fit(ipts, self.grid_shape)
        img = torch.stack([interp.predict(colors[:, i]) for i in range(3)])
        return FrameClass(img[None,:], interp.predict(self.pts[valid][:,2])[None,None,:], intrinsics=intrinsics).to(intrinsics.device)

    def pcl2open3d(self):
        import open3d
        pcd = open3d.geometry.PointCloud()
        #pcd.normals = open3d.utility.Vector3dVector(self.normals.cpu().numpy())
        pcd.points = open3d.utility.Vector3dVector(self.pts.cpu().numpy())
        pcd.colors = open3d.utility.Vector3dVector(self.colors.cpu().numpy())
        return pcd

    def to(self, d: Union[torch.device, torch.dtype]):
        self.radi = self.radi.to(d)
        self.normals = self.normals.to(d)
        self.gray = self.gray.to(d)
        self.kmat = self.kmat.to(d)
        self.opts = self.opts.to(d)
        self.pmat = self.pmat.to(d)
        self.device = self.opts.device
        return self
