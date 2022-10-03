from alley_oop.fusion.surfel_map import *
from alley_oop.network_core.raft.core.utils.flow_utils import remap_from_flow_nearest


class SurfelMapFlow(SurfelMap):
    def fuse(self, frame: FrameClass, pmat: torch.Tensor, flow: torch.Tensor, render_csp: torch.Tensor):
        assert pmat.shape == (4, 4)

        # prepare parameters
        self.img_shape = frame.shape
        kmat = self.kmat.clone()

        rgb = frame.img
        depth = frame.depth
        dtype = depth.dtype


        # prepare image and object coordinates
        ipts = create_img_coords_t(y=self.img_shape[-2], x=self.img_shape[-1]).to(
            dtype).to(self.device)

        # project depth to 3d-points in world coordinates
        opts = reverse_project(ipts=ipts, dpth=depth, rmat=pmat[:3, :3], tvec=pmat[:3, -1][..., None], kmat=kmat)

        # consider masked surfels and enforce channel x samples shape
        rgb = rgb.view(3, -1)

        # use optical flow to get correspondences (2D to 3D flow)
        vidx = render_csp.view(-1)
        n, _, h, w = flow.shape
        row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        flow_off_w = torch.round(flow[:, 0] + col_coords.to(flow.device))
        flow_off_h = torch.round(flow[:, 1] + row_coords.to(flow.device))
        midx = (w*flow_off_h + flow_off_w).long()

        valid = frame.mask & (flow_off_w >= 0) & (flow_off_w < w) & (flow_off_h >= 0) & (flow_off_h < h)
        vidx = vidx[valid.view(-1)]
        midx.view(-1)[~valid.view(-1)] = -1
        midx = (midx.view(-1)[valid.view(-1)]).long()

        # pre-select confidence elements
        conf = frame.confidence.view(1, -1) / self.conf_thr
        ccor = conf[:, midx]
        conf_idx = self.conf[:, vidx]

        # update existing points, intensities, normals, radii and confidences
        if self.average_points:
            self.opts[:, vidx] = (conf_idx * self.opts[:, vidx] + ccor * opts[:, midx]) / (conf_idx + ccor)
            self.rgb[:, vidx] = (conf_idx * self.rgb[:, vidx] + ccor * rgb[:, midx]) / (conf_idx + ccor)
        self.conf[:, vidx] = torch.clamp(conf_idx + ccor, 0.0, 1.0)  # saturate confidence to 1

        # create mask identifying unmatched indices
        mask = torch.ones(opts.shape[1], device=opts.device, dtype=torch.bool)
        mask[midx] = False
        # avoid fusion with invalid pixels
        mask &= frame.mask.view(-1)
        if self.dbug_opt:
            # print ratio of added points vs frame resolution
            ratio = mask.float().mean()
            print(ratio)

        # concatenate unmatched points, intensities, normals, radii and confidences
        self.opts = torch.cat((self.opts, self._downsample(opts)[:, mask]), dim=-1)
        self.rgb = torch.cat((self.rgb, self._downsample(rgb)[:, mask]), dim=-1)
        self.conf = torch.cat((self.conf, self._downsample(conf)[:, mask]), dim=-1)
        self.t_created = torch.cat((self.t_created, self.tick * torch.ones(1, mask.sum()).to(self.device)), dim=-1)

        self.tick = self.tick + 1

        # remove surfels
        self.remove_surfels_by_confidence_and_time()

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
        sort_idx = torch.argsort(self.conf, dim=1)[0]
        pts_h = torch.vstack([self.opts[:, sort_idx], torch.ones(self.opts.shape[1], dtype=self.opts.dtype, device=self.device)])
        npts, valid = forward_project2image(pts_h, img_shape=self.img_shape, kmat=intrinsics, rmat=extrinsics[:3, :3],
                               tvec=extrinsics[:3, -1][..., None])

        # generate sparse img maps and interpolate missing values
        img_coords = npts[1, valid].long(), npts[0, valid].long()
        confidence = torch.zeros(self.img_shape, dtype=self.opts.dtype, device=self.device)
        confidence[img_coords] = self.conf[0, sort_idx][valid]

        depth = torch.zeros(self.img_shape, dtype=self.opts.dtype, device=self.device)
        # confidence aware rendering. If multiple points project into the same pixel, we take a weighted sum of the values
        depth[img_coords] = self.opts[2, sort_idx][valid]
        mask = confidence[None,None,...] != 0.0
        depth = self.interpolate(depth[None,None,...])

        colors = torch.zeros((3, *self.img_shape), dtype=self.opts.dtype, device=self.device)
        colors[0][img_coords] = self.rgb[0, sort_idx][valid]
        colors[1][img_coords] = self.rgb[1, sort_idx][valid]
        colors[2][img_coords] = self.rgb[2, sort_idx][valid]
        colors = self.interpolate(colors[None,...])

        # relate 3D surfels with 2D points (used for optical flow based fusion)
        render_csp = -torch.ones(self.img_shape, dtype=torch.long, device=self.device)
        render_csp[img_coords] = sort_idx[valid]

        return FrameClass(colors, depth, intrinsics=intrinsics, mask=mask, confidence=confidence[None,None,...]).to(intrinsics.device), render_csp

    def _constructor(self):
        return SurfelMapFlow

    def remove_surfels_by_confidence_and_time(self):
        """ remove unstable points that have been created long time ago """

        ok_pts = ((self.conf >= 1.0) | ((self.tick - self.t_created) < self.t_max)).squeeze()
        self.opts = self.opts[:, ok_pts]
        self.rgb = self.rgb[:, ok_pts]
        self.conf = self.conf[:, ok_pts]
        self.t_created = self.t_created[:, ok_pts]
        return ok_pts