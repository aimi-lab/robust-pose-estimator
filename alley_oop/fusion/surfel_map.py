import torch

from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project, create_img_coords_t
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
        self.grid_shape = kwargs['grid_shape'] if 'grid_shape' in kwargs else self.gray.shape[-2:]

        # calculate depth points
        ipts = create_img_coords_t(y=self.grid_shape[-2], x=self.grid_shape[-1])
        self.opts = reverse_project(ipts=ipts, kmat=self.kmat, rmat=torch.eye(3), tvec=torch.zeros(3, 1), dpth=self.dept)

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
        
        # intialize tick as timestamp
        self.tick = 0
            
    def fuse(self, opts: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor = None, pmat: torch.Tensor = None):
        """
        https://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf
        """

        radi = (opts[:, 2] * 2**.5) / (self.flen * abs(normals[:, 2]))
        
        # find correspondence by projecting surfels to current frame
        vidx, idcs = self.match_surfels_by_projection(opts=opts, normals=normals, pmat=pmat)

        pcor, ncor, gcor, rcor = opts[idcs], normals[idcs], gray[idcs], radi[idcs]

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
        self.disp = torch.cat(self.disp, opts[~idcs])
        self.gray = torch.cat(self.gray, gcor[~idcs])
        self.radi = torch.cat(self.radi, radi[~idcs])
        self.normals = torch.cat(self.normals, normals[~idcs])

    def match_surfels_by_projection(self,
        opts: torch.Tensor = None,
        normals: torch.Tensor = None,
        pmat: torch.Tensor = None,
        d_thresh: float = 1, 
        n_thresh: float = 1,
        upsample_factor: int = 4,
        ):

        """
        https://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf
        """

        target_ipts = create_img_coords_t(y=self.grid_shape[-2]*upsample_factor, x=self.grid_shape[-2]*upsample_factor)
        global_ipts = forward_project(self.opts, kmat=self.kmat, rmat=pmat[:3, :3], tvec=pmat[:3, -1][:, None])

        # 0. exclude points outside field-of-view
        vidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & (global_ipts[0, :] < self.grid_shape[0]) & (global_ipts[1, :] < self.grid_shape[1])

        # valid projected points
        vpts = global_ipts[:, vidx]

        # quantize points (while considering super-sampling factor 4)
        global_ipts_quantized = torch.round(global_ipts[:, vidx]*upsample_factor)#/upsample_factor

        # associate global_ipts and target_ipts: H x W x Candidates
        get_arg = lambda el: torch.argwhere(torch.sum(torch.isin(global_ipts_quantized.T, el, assume_unique=True), dim=1) == 3)
        idx_list = [get_arg(el) for el in target_ipts.T]

        # another vectorized attempt
        # global_ipts_quantized[(global_ipts_quantized[0, :] == target_ipts[0, :]) & (global_ipts_quantized[1, :] == target_ipts[1, :])]

        # access global_opts belonging to an image index compute depth and normal differences

        # 1. depth distance constraint

        # 2. large normals constraint (20 degrees threshold)

        # 3. confidence constraint

        # 4. euclidean distance constraint


        return vidx, vpts
