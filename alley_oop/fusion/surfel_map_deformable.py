from alley_oop.fusion.surfel_map_flow import *
from alley_oop.utils.pytorch import MedianPool2d
from alley_oop.interpol.sparse_img_interpolation import SparseMedianInterpolator


class SurfelMapDeformable(SurfelMapFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warp_field = kwargs["warp_field"] if "warp_field" in kwargs else torch.zeros_like(self.opts)  # translational warp-field
        assert self.opts.shape == self.warp_field.shape
        self.median_filt = MedianPool2d(15, same=True, stride=1)
        self.interpolate = SparseMedianInterpolator(5)

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
        flow_trg_idx, flow_ref_idx, valid = self.get_flow_correspondences(frame, flow, render_csp)
        bidx = (global_ipts[0, :] >= 0) & (global_ipts[1, :] >= 0) & \
               (global_ipts[0, :] < self.img_shape[1]-1) & (global_ipts[1, :] < self.img_shape[0]-1)
        proj_trg_idx = self.get_match_indices(global_ipts[:, bidx])

        self.estimate_warpfield(opts, flow_ref_idx, flow_trg_idx, valid, global_ipts, bidx)
        # pre-select confidence elements
        conf = torch.ones_like(frame.confidence.view(1, -1)) / self.conf_thr

        # update confidences
        self.conf[:, flow_ref_idx[valid]] = torch.clamp(self.conf[:, flow_ref_idx[valid]] + conf[:, flow_trg_idx[valid]], 0.0, 1.0)  # saturate confidence to 1

        #self.remove_redundant_surfels(bidx, render_csp)

        # create mask identifying unmatched indices
        mask = torch.ones(opts.shape[1], device=opts.device, dtype=torch.bool)

        mask[flow_trg_idx[valid]] = False  # mask points with optical flow matches
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
        self.warp_field = transform[:3, :3] @ self.warp_field

    def transform_cpy(self, transform:torch.tensor):
        opts = transform[:3, :3] @ self.opts + transform[:3, 3, None]
        warp_field = transform[:3, :3] @ self.warp_field
        return self._constructor(opts=opts, kmat=self.kmat, rgb=self.rgb, img_shape=self.img_shape,
                                 depth_scale=self.depth_scale, conf=self.conf, warp_field=warp_field,
                                 normals=self.normals).to(self.device)

    def render(self, intrinsics: torch.tensor=None, extrinsics: torch.tensor=None, deform: bool=True):
        if deform:
            d = self.deform_cpy()
            return d.render(intrinsics, extrinsics, deform=False)
        else:
            return super().render(intrinsics, extrinsics)

    def estimate_warpfield(self, opts, flow_ref_idx, flow_trg_idx, valid, ipts, bidx):
        # use the residuals as a 3d translational warp-field
        # the assumptions are:
        # 1: small residuals are due to depth, flow or pose estimation errors and should be ignored
        # 2: large residuals are mainly due to deformations
        # -> no accumulation of small drift errors, but update of scene when deformations are significant
        warp = opts[:, flow_trg_idx].view(3,*self.img_shape) - self.opts[:, flow_ref_idx].view(3,*self.img_shape)
        warp[:, ~valid.view(self.img_shape)] = 0.0
        warp = self.median_filt(warp.unsqueeze(0)).squeeze()

        self.warp_field = torch.zeros_like(self.opts)
        # use projective association to interpolate warp-field for not rendered points
        self.warp_field[0, bidx] = warp[0][(ipts[1, bidx].long(), ipts[0, bidx].long())]
        self.warp_field[1, bidx] = warp[1][(ipts[1, bidx].long(), ipts[0, bidx].long())]
        self.warp_field[2, bidx] = warp[2][(ipts[1, bidx].long(), ipts[0, bidx].long())]
        not_deformed = torch.sum(self.warp_field ** 2, dim=0) <= self.d_thresh ** 2
        self.warp_field[:, not_deformed] = 0.0
        return self.warp_field

    def remove_redundant_surfels(self, in_fov, rendered):
        # remove points that are in the FOV but not rendered
        mask = torch.zeros(self.opts.shape[1], dtype=torch.bool, device=self.opts.device)
        mask[~in_fov] = True
        mask[rendered] = True
        self.opts = self.opts[:, mask]
        self.rgb = self.rgb[:, mask]
        self.conf = self.conf[:, mask]
        self.t_created = self.t_created[:, mask]
        self.warp_field = self.warp_field[:, mask]

    def get_flow_correspondences(self, frame, flow, render_csp):
        n, _, h, w = flow.shape
        row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        flow_off_w = torch.round(flow.squeeze()[0] + col_coords.to(flow.device))
        flow_off_h = torch.round(flow.squeeze()[1] + row_coords.to(flow.device))
        midx = (w * flow_off_h + flow_off_w).long()
        valid = frame.mask.squeeze() & (flow_off_w >= 0) & (flow_off_w < w) & (flow_off_h >= 0) & (flow_off_h < h)
        midx[~valid] = 0
        return midx.long().view(-1), render_csp.view(-1), valid.view(-1)

    def remove_surfels_by_confidence_and_time(self):
        """ remove unstable points that have been created long time ago """
        ok_pts = super().remove_surfels_by_confidence_and_time()
        self.warp_field = self.warp_field[:, ok_pts]
        return ok_pts
