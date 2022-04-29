import torch

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
        self.dept = kwargs['dept'] if 'dept' in kwargs else torch.Tensor()
        self.gray = kwargs['gray'] if 'gray' in kwargs else torch.Tensor()
        self.pmat = kwargs['pmat'] if 'pmat' in kwargs else torch.eye(4)    # extrinsics
        self.kmat = kwargs['kmat'] if 'kmat' in kwargs else torch.eye(3)    # intrinsics
        self.normals = kwargs['normals'] if 'normals' in kwargs else torch.Tensor()
        self.img_shape = kwargs['img_shape'] if 'img_shape' in kwargs else None

        # calculate object points
        if self.dept.numel() > 0 and self.img_shape is not None:
            ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1])
            self.opts = reverse_project(ipts=ipts, kmat=self.kmat, rmat=torch.eye(3), tvec=torch.zeros(3, 1), dpth=self.dept.reshape(self.img_shape))
        elif self.dept.numel() == 0 and self.opts.numel() > 0 and self.img_shape is not None:
            # rotate, translate and forward-project points
            npts = forward_project(self.opts, kmat=self.kmat, rmat=self.pmat[:3, :3], tvec=self.pmat[:3, -1][..., None], inhomogenize_opt=True)
            self.dept = img_map_torch(img=npts[2].reshape(self.img_shape), npts=npts, mode='bilinear')
        elif self.img_shape is None and self.dept.numel() == 0 and self.opts.numel() == 0:
            raise BaseException('Image shape must be provided if depth or objects are missing')

        # initiliaze focal length
        self.flen = (self.kmat[0, 0] + self.kmat[1, 1]) / 2

        # initialize radii
        self.radi = torch.Tensor()
        if self.dept.numel() > 0 and self.normals.numel() == self.dept.numel():
            self.radi = (self.disp[:, 2] * 2**.5) / (self.flen * abs(self.normals[:, 2]))
        elif self.opts.numel() > 0:
            self.radi = torch.ones((1, self.opts.shape[1]))

        # initiliaze confidence
        gamma = self.dept.flatten()/torch.max(self.dept)
        self.conf = torch.exp(-.5 * gamma**2 / .6**2)

        # upsample value
        self.upscale = 1    # TODO: enable value other than 1

        # intialize tick as timestamp
        self.tick = 0
            
    def fuse(self, dept: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor = None, pmat: torch.Tensor = None):

        # compute opts considering upsampling
        ipts = create_img_coords_t(y=self.img_shape[-2]*self.upscale, x=self.img_shape[-1]*self.upscale)
        dept = torch.nn.functional.upsample(dept, scale_factor=self.upscale, mode='bilinear', align_corners=None)
        opts = reverse_project(ipts=ipts, dpth=dept, rmat=self.pmat[:3, :3], tvec=self.pmat[:3, -1][..., None], kmat=self.kmat)

        if normals is None:
            # TODO: ensure normals have same length as opts etc.
            normals = normals_from_regular_grid(opts.reshape((self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale, 3)))
        
        # enforce channel x samples shape
        normals = normals.reshape(3, -1)
        gray = gray.flatten()[None, :]
        dept = dept.flatten()[None, :]

        # initialize global points
        global_ipts = forward_project(self.opts, kmat=self.kmat, rmat=pmat[:3, :3], tvec=pmat[:3, -1][:, None])
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[0]) & (global_ipts[1, :] < self.img_shape[1])

        # find correspondence by projecting surfels to current frame
        midx = self.get_match_indices(global_ipts[:, bidx])           # image border constraints
        fidx = self.filter_points_by_comparison(opts=opts, normals=normals, midx=midx, vidx=bidx)
        #fidx = self.remove_duplicates(opts=opts, pmat=pmat, vidx=fidx)     
        midx = self.get_match_indices(global_ipts[:, bidx][:, fidx])  # filter constraints

        # compute radii
        radi = (opts[2, :] * 2**.5) / (self.flen * abs(normals[2, :]))[None, :]

        # select corresponding elements
        ocor, ncor, gcor, rcor = opts[:, midx], normals[:, midx], gray[:, midx], radi[:, midx]

        # assign confidence
        gamma = ocor[2, :]/torch.max(ocor[2, :])
        cora = torch.exp(-.5 * gamma**2 / .6**2)

        # update existing points, intensities, normals, radii and confidences
        self.opts[:, bidx][:, fidx] = self.conf[bidx][fidx]*self.opts[:, bidx][:, fidx] + cora*ocor / (self.conf[bidx][fidx] + cora)
        self.gray[:, bidx][:, fidx] = self.conf[bidx][fidx]*self.gray[:, bidx][:, fidx] + cora*gcor / (self.conf[bidx][fidx] + cora)
        self.radi[:, bidx][:, fidx] = self.conf[bidx][fidx]*self.radi[:, bidx][:, fidx] + cora*rcor / (self.conf[bidx][fidx] + cora)
        self.normals[:, bidx][:, fidx] = self.conf[bidx][fidx]*self.normals[:, bidx][:, fidx] + cora*ncor / (self.conf[bidx][fidx] + cora)
        self.conf[bidx][fidx] = self.conf[bidx][fidx] + cora

        # concatenate unmatched points, intensities, normals, radii and confidences
        mask = torch.ones(opts.shape[1], dtype=bool)
        mask[midx.unique()] = False
        self.opts = torch.cat((self.opts, opts[:, mask]), dim=-1)
        self.gray = torch.cat((self.gray, gray[:, mask]), dim=-1)
        self.radi = torch.cat((self.radi, radi[:, mask]), dim=-1)
        self.normals = torch.cat((self.normals, normals[:, mask]), dim=-1)

        self.tick = self.tick + 1

    def get_match_indices(
        self,
        ipts: torch.Tensor = None,
        ):

        # quantize points (while considering super-sampling factor)
        ipts_quantized = torch.round(ipts*self.upscale)

        # get point correspondence from indexing as flattened 2D indices
        midx = ipts_quantized[1, :] * self.img_shape[0] * self.upscale + ipts_quantized[0, :]

        return midx.long()
    
    def filter_points_by_comparison(
        self,
        opts: torch.Tensor = None,
        normals: torch.Tensor = None,
        midx: torch.Tensor = None,
        vidx: torch.Tensor = None,
        d_thresh: float = 1, 
        n_thresh: float = 20,
        ):

        # 1. depth distance constraint
        didx = abs(opts[2, midx] - self.opts[2, vidx]) < d_thresh

        # 2. normals constraint (20 degrees threshold)
        nidx = batched_dot_product(normals.T[midx], self.normals.T[vidx]) < n_thresh/180*torch.pi

        # 3. confidence constraint
        #TODO

        # 4. combine constraints
        fidx = vidx[vidx.clone()] & didx & nidx

        return fidx

    def remove_duplicates(
        self,
        opts: torch.Tensor,
        pmat: torch.Tensor,
        vidx: torch.Tensor,
        ):
        """
        #TODO
        """
        
        # identify duplicates to enforce unique correspondence assignment 
        global_ipts = forward_project(self.opts, kmat=self.kmat, rmat=pmat[:3, :3], tvec=pmat[:3, -1][:, None])
        global_ipts_quantized = torch.round(global_ipts[:, vidx]*self.upscale)
        gvec = global_ipts_quantized[1, :] * self.img_shape[0] * self.upscale + global_ipts_quantized[0, :]
        oidx, bins = torch.unique(gvec, sorted=True, return_counts=True)
        duplicates = oidx[bins > 1]

        global_opts = self.opts[:, vidx]
        for d in duplicates:
            cidx = gvec == d
            candidates  = global_opts[:, cidx]
            dist = torch.sum((opts[:, d.long()][:, None] - candidates)**2, dim=0)**.5
            global_opts.pop()
