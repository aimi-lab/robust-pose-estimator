from alley_oop.fusion.surfel_map_flow import *


class SurfelMapDeformable(SurfelMapFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warp_field = kwargs["warp_field"] if "warp_field" in kwargs else torch.zeros_like(self.opts)  # translational warp-field
        assert self.opts.shape == self.warp_field.shape

    def fuse(self, *args):
        frame, pmat, flow, render_csp = args
        assert pmat.shape == (4, 4)

        # prepare parameters
        self.img_shape = frame.shape
        kmat = self.kmat.clone()
        pmat_inv = torch.linalg.inv(pmat)

        rgb = frame.img
        depth = frame.depth
        dtype = depth.dtype

        # prepare image and object coordinates
        ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1]).to(
            dtype).to(self.device)

        # project depth to 3d-points in world coordinates
        opts = reverse_project(ipts=ipts, dpth=depth, rmat=pmat[:3, :3], tvec=pmat[:3, -1][..., None], kmat=kmat)

        # project surfel map to current image plane
        opts_def = self._deform(self.opts, self.warp_field)
        global_ipts = forward_project(opts_def, kmat=kmat, rmat=pmat_inv[:3, :3], tvec=pmat_inv[:3, -1][:, None])

        # consider masked surfels and enforce channel x samples shape
        rgb = rgb.view(3, -1)

        # use optical flow to get correspondences (2D to 3D flow)
        flow_trg_idx, flow_ref_idx = self.get_flow_correspondences(frame, flow, render_csp)
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & \
               (global_ipts[0, :] < self.img_shape[1]-1) & (global_ipts[1, :] < self.img_shape[0]-1)
        proj_trg_idx = self.get_match_indices(global_ipts[:, bidx])

        # filter with respect to 3d distance
        dists = opts[:, flow_trg_idx] - self.opts[:, flow_ref_idx]
        valid = torch.sum(dists**2, dim=0) < self.d_thresh**2
        bidx = flow_trg_idx[valid]
        flow_ref_idx = flow_ref_idx[valid]

        # update warp-field (this is a very naive approach simply taking the residuals as the new warp-field)
        self.warp_field = torch.zeros_like(self.opts)
        self.warp_field[:, flow_ref_idx] = dists
        # pre-select confidence elements
        conf = frame.confidence.view(1, -1) / self.conf_thr

        # update existing points, intensities, normals, radii and confidences
        ccor = conf[:, bidx]
        conf_idx = self.conf[:, flow_ref_idx]
        if self.average_points:
            self.opts[:, flow_ref_idx] = (conf_idx * self.opts[:, flow_ref_idx] + ccor * opts[:, bidx]) / (conf_idx + ccor)
            self.rgb[:, flow_ref_idx] = (conf_idx * self.rgb[:, flow_ref_idx] + ccor * rgb[:, bidx]) / (conf_idx + ccor)
        self.conf[:, flow_ref_idx] = torch.clamp(conf_idx + ccor, 0.0, 1.0)  # saturate confidence to 1

        # create mask identifying unmatched indices
        mask = torch.ones(opts.shape[1], device=opts.device, dtype=torch.bool)

        mask[flow_trg_idx] = False  # mask points with optical flow matches
        mask[proj_trg_idx] = False  # mask points with projective associations
        # avoid fusion with invalid pixels
        mask &= frame.mask.view(-1)
        if self.dbug_opt:
            # print ratio of added points vs frame resolution
            ratio = mask.float().mean()
            print(ratio)

        # concatenate unmatched points, intensities, normals, radii and confidences
        self.opts = torch.cat((self.opts, opts[:, mask]), dim=-1)
        self.rgb = torch.cat((self.rgb, rgb[:, mask]), dim=-1)
        self.conf = torch.cat((self.conf, conf[:, mask]), dim=-1)
        self.t_created = torch.cat((self.t_created, self.tick * torch.ones(1, mask.sum()).to(self.device)), dim=-1)

        self.tick = self.tick + 1

        # remove surfels
        self.remove_surfels_by_confidence_and_time()

    @property
    def _constructor(self):
        return SurfelMapDeformable

    def inv_deform(self, warp_field: torch.Tensor = None):
        """
        deform surfels with inverted warp_field inplace
        :param warp_field: nx3 translational warp-field (optional)
        """
        warp_field = self.warp_field if warp_field is None else warp_field
        self.deform(-warp_field)

    def inv_deform_cpy(self, warp_field: torch.Tensor = None):
        """
        deform surfels with inverted warp_field inplace
        :param warp_field: nx3 translational warp-field (optional)
        """
        warp_field = self.warp_field if warp_field is None else warp_field
        return self.deform_cpy(-warp_field)

    def _deform(self, opts, warp_field: torch.Tensor = None):
        return opts + warp_field.T

    def deform(self, warp_field: torch.Tensor = None):
        """
        deform surfels with warp_field inplace
        :param warp_field: nx3 translational warp-field (optional)
        """
        warp_field = self.warp_field if warp_field is None else warp_field
        self.opts = self._deform(self.opts, warp_field)
        self.warp_field = warp_field

    def deform_cpy(self, warp_field: torch.Tensor = None):
        """
        deform surfels with warp_field and return a copy
        :param warp_field: nx3 translational warp-field (optional)
        """
        warp_field = self.warp_field if warp_field is None else warp_field
        opts = self.opts.clone()
        opts = self._deform(opts, warp_field)
        map = self._constructor(opts=opts, kmat=self.kmat, gray=self.gray, img_shape=self.img_shape,
                                  depth_scale=self.depth_scale, conf=self.conf, warp_field=warp_field,
                                  active=self.active).to(self.device)

        return map

    def transform(self, transform:torch.tensor):
        self.opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        self.warp_field = transform[:3, :3] @ self.warp_field + transform[:3, 3, None]

    def transform_cpy(self, transform:torch.tensor):
        opts = transform[:3, :3] @ self.opts + transform[:3, 3, None]
        warp_field = transform[:3, :3] @ self.warp_field + transform[:3, 3, None]
        return self._constructor(opts=opts, kmat=self.kmat, rgb=self.rgb, img_shape=self.img_shape,
                                 depth_scale=self.depth_scale, conf=self.conf, warp_field=warp_field).to(self.device)

    def render(self, intrinsics: torch.tensor=None, extrinsics: torch.tensor=None, deform: bool=False):
        if deform:
            d = self.deform_cpy()
        else:
            d = self
        return d.render(intrinsics, extrinsics)
