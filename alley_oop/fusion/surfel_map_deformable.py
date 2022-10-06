from alley_oop.fusion.surfel_map_flow import *
from alley_oop.interpol.gp_warpfield import GP_WarpFieldEstimator


class SurfelMapDeformable(SurfelMapFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warp_field = kwargs["warp_field"] if "warp_field" in kwargs else torch.zeros_like(self.opts)  # translational warp-field
        assert self.opts.shape == self.warp_field.shape
        self.warp_field_estimator = GP_WarpFieldEstimator(length_scale=0.1,
                                                          noise_level=0.001)
        self.n_samples = 256

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
        flow_trg_idx, flow_ref_idx, flow_trg_idx_lr, flow_ref_idx_lr = self.get_flow_correspondences(frame, flow, render_csp)
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & \
               (global_ipts[0, :] < self.img_shape[1]-1) & (global_ipts[1, :] < self.img_shape[0]-1)
        proj_trg_idx = self.get_match_indices(global_ipts[:, bidx])

        # use the residuals as a 3d translational warp-field
        # the assumptions are:
        # 1: small residuals are due to depth, flow or pose estimation errors and should be ignored
        # 2: large residuals are mainly due to deformations
        # -> no accumulation of small drift errors, but update of scene when deformations are significant

        # fit and predict warp-field for canonical model to current frame
        # we need to subsample to condition the GP for computational reasons, then we can evaluate it on all points.
        self.warp_field_estimator.fit(self.opts[:, flow_ref_idx_lr], opts[:, flow_trg_idx_lr])

        # estimate warp-field from canonical to current frame (with extrapolation), we only deform active points for computational reasons
        self.warp_field = self.warp_field_estimator.predict(self.opts)
        # threshold warp-field for small deformations to avoid drift
        not_deformed = torch.sum(self.warp_field ** 2, dim=0) <= self.d_thresh ** 2
        self.warp_field[:, not_deformed] = 0.0

        # pre-select confidence elements
        conf = frame.confidence.view(1, -1) / self.conf_thr

        # update confidences
        self.conf[:, flow_ref_idx] = torch.clamp(self.conf[:, flow_ref_idx] + conf[:, flow_trg_idx], 0.0, 1.0)  # saturate confidence to 1

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
        self.opts = torch.cat((self.opts, opts[:, mask]), dim=-1) #We should deform points before adding locations
        self.rgb = torch.cat((self.rgb, rgb[:, mask]), dim=-1)
        self.conf = torch.cat((self.conf, conf[:, mask]), dim=-1)
        self.t_created = torch.cat((self.t_created, self.tick * torch.ones((1, mask.sum()), device=self.device)), dim=-1)
        self.warp_field = torch.cat((self.warp_field,  torch.zeros((3, mask.sum()), device=self.device)), dim=-1)
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
        return opts + warp_field

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
        map = self._constructor(opts=opts, kmat=self.kmat, rgb=self.rgb, img_shape=self.img_shape,
                                  depth_scale=self.depth_scale, conf=self.conf, warp_field=warp_field,
                                  normals=self.normals).to(self.device)

        return map

    def transform(self, transform:torch.tensor):
        self.opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        self.warp_field = transform[:3, :3] @ self.warp_field + transform[:3, 3, None]

    def transform_cpy(self, transform:torch.tensor):
        opts = transform[:3, :3] @ self.opts + transform[:3, 3, None]
        warp_field = transform[:3, :3] @ self.warp_field + transform[:3, 3, None]
        return self._constructor(opts=opts, kmat=self.kmat, rgb=self.rgb, img_shape=self.img_shape,
                                 depth_scale=self.depth_scale, conf=self.conf, warp_field=warp_field,
                                 normals=self.normals).to(self.device)

    def render(self, intrinsics: torch.tensor=None, extrinsics: torch.tensor=None, deform: bool=True):
        if deform:
            d = self.deform_cpy()
            return d.render(intrinsics, extrinsics, deform=False)
        else:
            return super().render(intrinsics, extrinsics)

    def get_flow_correspondences(self, frame, flow, render_csp, step=32):
        n, _, h, w = flow.shape
        vidx = render_csp.view(-1)
        row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        flow_off_w = torch.round(flow.squeeze()[0] + col_coords.to(flow.device))
        flow_off_h = torch.round(flow.squeeze()[1] + row_coords.to(flow.device))
        midx = (w * flow_off_h + flow_off_w).long()
        valid = frame.mask.squeeze() & (flow_off_w >= 0) & (flow_off_w < w) & (flow_off_h >= 0) & (flow_off_h < h)

        midx_lowres = midx[::step, ::step]
        midx_lowres = midx_lowres[valid[::step, ::step]].view(-1).long()
        vidx_lowres = render_csp[::step, ::step]
        vidx_lowres = vidx_lowres[valid[::step, ::step]].view(-1)

        vidx = vidx[valid.view(-1)]
        midx = (midx.view(-1)[valid.view(-1)]).long()
        return midx, vidx ,midx_lowres, vidx_lowres
