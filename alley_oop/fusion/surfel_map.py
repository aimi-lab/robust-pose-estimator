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

        # initiliaze focal length
        self.flen = (self.kmat[0, 0] + self.kmat[1, 1]) / 2

        # initialize radii
        self.radi = torch.Tensor()
        if self.dept.numel() > 0 and self.normals.numel() == self.dept.numel():
            self.radi = (self.disp[:, 2] * 2**.5) / (self.flen * abs(self.normals[:, 2]))
        elif self.opts.numel() > 0:
            self.radi = torch.ones((1, self.opts.shape[1]))

        # initialize confidence
        self.conf = torch.Tensor()
        if self.opts.numel() > 0:
            gamma = self.opts[2].flatten()/torch.max(self.opts[2])
            self.conf = torch.exp(-.5 * gamma**2 / .6**2)[None ,:]

        # upsample value
        self.upscale = 1    # TODO: enable value other than 1

        # intialize tick as timestamp
        self.tick = 0
            
    def fuse(self, dept: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor = None, pmat: torch.Tensor = None):
        
        # update image shape
        self.img_shape = gray.shape[-2:] if self.img_shape is None else self.img_shape

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

        # project all surfels to current image frame
        global_ipts = forward_project(self.opts, kmat=self.kmat, rmat=pmat[:3, :3], tvec=pmat[:3, -1][:, None])
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[1]) & (global_ipts[1, :] < self.img_shape[0])
        
        # test global point projection
        #midx = self.get_match_indices(global_ipts[:, 100:])  
        #gpts = global_ipts[:, 100:][:, midx]
        #timg = img_map_torch(img=gpts[2].reshape(self.img_shape)[None, None, ...], npts=gpts)
        #import matplotlib.pyplot as plt
        #plt.imshow(timg.cpu().numpy()[0,0, ...])
        #plt.show()

        # find correspondence by projecting surfels to current frame
        midx = self.get_match_indices(global_ipts[:, bidx])           # image border constraints
        fidx = self.filter_points_by_comparison(opts=opts, normals=normals, midx=midx, vidx=bidx)
        kidx = self.remove_duplicates(opts=opts, vidx=fidx)     
        midx = self.get_match_indices(global_ipts[:, bidx][:, fidx])  # filter constraints

        # compute radii
        radi = (opts[2, :] * 2**.5) / (self.flen * abs(normals[2, :]))[None, :]

        # compute confidence
        gamma = opts[2, :]/torch.max(opts[2, :])
        conf = torch.exp(-.5 * gamma**2 / .6**2)[None, :]

        # select corresponding elements
        ocor, ncor, gcor, rcor, ccor = opts[:, midx], normals[:, midx], gray[:, midx], radi[:, midx], conf[:, midx]

        # update existing points, intensities, normals, radii and confidences
        conf_idx = self.conf[:, bidx][:, fidx][:, midx]
        self.opts[:, bidx][:, fidx][:, midx] = conf_idx*self.opts[:, bidx][:, fidx] + ccor*ocor / (conf_idx + ccor)
        self.gray[:, bidx][:, fidx][:, midx] = conf_idx*self.gray[:, bidx][:, fidx] + ccor*gcor / (conf_idx + ccor)
        self.radi[:, bidx][:, fidx][:, midx] = conf_idx*self.radi[:, bidx][:, fidx] + ccor*rcor / (conf_idx + ccor)
        self.normals[:, bidx][:, fidx][:, midx] = conf_idx*self.normals[:, bidx][:, fidx] + ccor*ncor / (conf_idx + ccor)
        self.conf[:, bidx][:, fidx][:, midx] = conf_idx + ccor

        # concatenate unmatched points, intensities, normals, radii and confidences
        mask = torch.ones(opts.shape[1], dtype=bool)
        mask[midx.unique()] = False
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
        didx = abs(opts[2] - self.opts[2, vidx][midx]) < d_thresh

<<<<<<< HEAD
        # 2. normals angle deviation constraint (20 degrees threshold by default)
        nidx = batched_dot_product(normals.T[midx], self.normals.T[vidx]) < n_thresh/180*torch.pi
=======
        # 2. normals constraint (20 degrees threshold)
        nidx = batched_dot_product(normals.T[midx], self.normals.T[vidx]) > torch.cos(n_thresh/180*torch.pi)
>>>>>>> 854c0155c37580a14289134cd8491295fd21bda9

        # combine constraints
        fidx = vidx[vidx.clone()] & didx & nidx

        return fidx

    def remove_duplicates(
        self,
        opts: torch.Tensor,
        midx: torch.Tensor,
        vidx: torch.Tensor = None,
        ):
        """
        remove points mapping to the same pixel location to enforce unique correspondence assignment 
        """
        
        # identify duplicates 
        a = torch.unique(midx, return_inverse=True, sorted=False, return_counts=True)
        oidx = a[0][a[1]]
        bins = a[2][a[1]]
        duplicates = oidx[bins > 1].long()

        vidx = torch.ones(self.opts.shape[1], dtype=bool) if vidx is None else vidx
        kidx = torch.ones(self.opts.shape[1], dtype=bool)
        for d in duplicates:
            # 3. confidence constraint
            candidates = self.conf[:, vidx][:, midx == d]
            # 4. euclidean distance constraint (if necessary)
            if torch.sum(candidates == torch.min(candidates)) > 1:
                candidates = torch.sum((opts[:, d][:, None] - self.opts[:, vidx][:, midx == d])**2, dim=0)**.5
            if torch.sum(candidates == torch.min(candidates)) > 1:
                candidates = torch.arange(len(candidates))
            mask = torch.ones(self.opts.shape[1], dtype=bool)
            mask[torch.where(midx==d)[0][torch.argmin(candidates)]] = False
            kidx[(midx == d) & mask] = 0

        return kidx
