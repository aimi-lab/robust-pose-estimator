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
        elif self.dept.numel() > 0:
            self.radi = torch.ones(self.dept.numel())

        # initiliaze confidence
        gamma = self.dept.flatten()/torch.max(self.dept)
        self.conf = torch.exp(-.5 * gamma**2 / .6**2)

        # upsample value
        self.upscale = 2

        # intialize tick as timestamp
        self.tick = 0
            
    def fuse(self, dept: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor = None, pmat: torch.Tensor = None):

        # compute opts considering upsampling
        ipts = create_img_coords_t(y=self.img_shape[-2]*self.upscale, x=self.img_shape[-1]*self.upscale)
        dept = torch.nn.functional.upsample(dept, scale_factor=self.upscale, mode='bilinear', align_corners=None)
        opts = reverse_project(ipts=ipts, dpth=dept, rmat=self.pmat[:3, :3], tvec=self.pmat[:3, -1][..., None], kmat=self.kmat)
        normals = normals_from_regular_grid(opts.reshape((self.img_shape[0]*self.upscale, self.img_shape[1]*self.upscale, 3)))

        radi = (opts[:, 2] * 2**.5) / (self.flen * abs(normals[:, 2]))
        
        # find correspondence by projecting surfels to current frame
        midx, vidx = self.match_surfels_by_projection(pmat=pmat)
        fidx = self.filter_by_comparison(opts=opts, normals=normals, midx=midx, vidx=vidx)

        pcor, ncor, gcor, rcor = opts[fidx], normals[fidx], gray[fidx], radi[fidx]

        # assign confidence
        gamma = pcor[:, 2]/torch.max(pcor[:, 2])
        cora = torch.exp(-.5 * gamma**2 / .6**2)

        # update existing points, normals and confidences
        self.disp[vidx] = self.conf*self.disp[vidx] + cora*pcor / (self.conf + cora)
        self.gray[vidx] = self.conf*self.gray[vidx] + cora*gcor / (self.conf + cora)
        self.radi[vidx] = self.conf*self.radi[vidx] + cora*rcor / (self.conf + cora)
        self.normals[vidx] = self.conf*self.normals[vidx] + cora*ncor / (self.conf + cora)
        self.conf[vidx] = self.conf + cora
        self.tick = self.tick + 1

        # concatenate unmatched points, normals and confidences
        self.disp = torch.cat((self.disp, opts[~fidx]), dim=-1)
        self.gray = torch.cat((self.gray, gcor[~fidx]), dim=-1)
        self.radi = torch.cat((self.radi, radi[~fidx]), dim=-1)
        self.normals = torch.cat((self.normals, normals[~fidx]), dim=-1)

    def match_surfels_by_projection(
        self,
        pmat: torch.Tensor = None,
        ):

        #target_ipts = create_img_coords_t(y=self.img_shape[-2]*upsample_factor, x=self.img_shape[-1]*upsample_factor)
        #target_ipts = torch.dstack(torch.meshgrid(torch.arange(self.img_shape[-2]), torch.arange(self.img_shape[-1])))
        global_ipts = forward_project(self.opts, kmat=self.kmat, rmat=pmat[:3, :3], tvec=pmat[:3, -1][:, None])

        # 0. exclude points outside field-of-view
        vidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.img_shape[0]) & (global_ipts[1, :] < self.img_shape[1])

        # quantize points (while considering super-sampling factor 4)
        global_ipts_quantized = torch.round(global_ipts[:, vidx]*self.upscale)

        # flatten points
        ivec = torch.arange(self.img_shape[-2]*self.img_shape[-1]*self.upscale**2)
        gvec = global_ipts_quantized[1, :] * self.img_shape[0] * self.upscale + global_ipts_quantized[0, :]

        # get correspondence from indexing as flattened 2D indices
        midx = ivec[gvec.long()]

        # associate global_ipts and target_ipts: H x W x Candidates
        #y = midx % (self.img_shape[1] * upsample_factor)
        #x = midx // (self.img_shape[1] * upsample_factor)

        return midx, vidx
    
    def filter_by_comparison(
        self,
        opts: torch.Tensor = None,
        normals: torch.Tensor = None,
        midx: torch.Tensor = None,
        vidx: torch.Tensor = None,
        d_thresh: float = 1, 
        n_thresh: float = 1,
        ):

        # 1. depth distance constraint
        didx = abs(opts[2, midx] - self.opts[2, vidx]) < d_thresh

        # 2. normals constraint (20 degrees threshold)
        nidx = (normals.T.reshape(-1, 3)[midx] @ self.normals[:, vidx]) > n_thresh

        # 3. confidence constraint

        # 4. euclidean distance constraint


        fidx = vidx[vidx.clone()] & didx & nidx

        return fidx
