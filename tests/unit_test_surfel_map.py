import unittest
from pathlib import Path

import torch
import imageio
import matplotlib.pyplot as plt

from alley_oop.utils.pfm_handler import load_pfm
from alley_oop.geometry.pinhole_transforms import reverse_project, forward_project, create_img_coords_t, disp2depth
from alley_oop.geometry.lie_3d import lie_se3_to_SE3
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.utils.rgb2gray import rgb2gray_t
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.interpol.synth_view import synth_view
from alley_oop.interpol.synth_depth import synth_depth


class SurfelMapTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SurfelMapTest, self).__init__(*args, **kwargs)

    def setUp(self):

        self.plt_opt = True

        self.kmat = torch.eye(3)
        self.pmat = torch.eye(4)

        self.data_path = Path.cwd() / 'tests' / 'test_data'
        self.disp = load_pfm(self.data_path / '000000l.pfm')[0]
        self.limg = imageio.imread(self.data_path / '000000l.png')

        # pseudo camera
        self.kmat[0, 0] = 1000
        self.kmat[1, 1] = 1000
        self.kmat[0, 2] = self.limg.shape[1] // 2
        self.kmat[1, 2] = self.limg.shape[0] // 2

    def test_point_fusion(self):

        # crop image parts
        self.gtruth_disp = torch.tensor(self.disp.copy())[None, None, ...]
        self.gtruth_limg = torch.from_numpy(self.limg.copy()[None, ...]).permute(0, 3, 1, 2)

        # convert to gray
        self.gtruth_gray = rgb2gray_t(self.gtruth_limg, ax0=1)

        # convert disparity to depth
        self.gtruth_dept = disp2depth(self.gtruth_disp, kmat=self.kmat).reshape(self.gtruth_disp.shape)

        # create image coordinates
        grid_shape = self.gtruth_limg.shape[-2:]
        ipts = create_img_coords_t(y=grid_shape[-2], x=grid_shape[-1])

        # project to space
        self.gtruth_opts = reverse_project(ipts=ipts, kmat=self.kmat, rmat=torch.eye(3), tvec=torch.zeros(3, 1), dpth=self.gtruth_dept)

        # compute normals
        self.gtruth_normals = normals_from_regular_grid(self.gtruth_opts.reshape((*grid_shape, 3)))

        # separate into global map and target_map
        gap = 100
        self.kmat[:2, -1] -= gap/2
        self.target_opts = self.gtruth_opts.reshape(3, *grid_shape)[..., gap:-gap, gap:-gap].reshape(3, -1)
        self.target_gray = self.gtruth_gray[..., gap:-gap, gap:-gap]
        self.target_dept = self.gtruth_dept[..., gap:-gap, gap:-gap]
        self.target_normals = self.gtruth_normals[gap+1:-gap-1, gap+1:-gap-1, :]
        self.global_opts = self.gtruth_opts.reshape(3, *grid_shape)[..., 2*gap:, 2*gap:].reshape(3, -1)
        self.global_gray = self.gtruth_gray[..., 2*gap:, 2*gap:].reshape(-1)
        self.global_dept = self.gtruth_dept[..., 2*gap:, 2*gap:].reshape(-1)
        self.global_normals = self.gtruth_normals[2*gap-1:, 2*gap-1:].reshape(3, -1)

        # break uniqueness and order in global points
        shuffle_idx = torch.randperm(self.global_opts.shape[1])
        self.global_opts = torch.cat((self.global_opts[:, :gap], self.global_opts[:, shuffle_idx]), dim=-1)
        self.global_gray = torch.cat((self.global_gray[:gap], self.global_gray[shuffle_idx]), dim=-1)
        self.global_dept = torch.cat((self.global_dept[:gap], self.global_dept[shuffle_idx]), dim=-1)
        self.global_normals = torch.cat((self.global_normals[:, :gap], self.global_normals[:, shuffle_idx]), dim=-1)

        # pseudo pose deviation
        torch.manual_seed(3008)
        self.pmat = lie_se3_to_SE3(pvec=torch.randn(6)*1e-6)
        ipts = create_img_coords_t(y=self.target_dept.shape[-2], x=self.target_dept.shape[-1])
        self.mockup_opts = reverse_project(ipts=ipts, kmat=self.kmat, rmat=self.pmat[:3, :3], tvec=self.pmat[:3, -1][:, None], dpth=self.target_dept)

        # initialize surfel map
        surf_map = SurfelMap(opts=self.global_opts, dept=self.global_dept, gray=self.global_gray, normals=self.global_normals, pmat=torch.eye(4), kmat=self.kmat)
        surf_map.img_shape = self.target_gray.shape[-2:]

        # update surfel map
        surf_map.fuse(dept=self.target_dept, gray=self.target_gray, normals=self.target_normals, pmat=torch.eye(4))

        # test assertions
        self.assertTrue(surf_map.opts.numel() > self.global_opts.numel(), 'Number of surfel map points too little')

        if self.plt_opt:
            from mayavi import mlab
            from alley_oop.utils.mlab_plot import mlab_rgbd
            ds = 8
            gpts = self.global_opts.cpu().numpy()[:, ::ds]
            gimg = self.global_gray.permute(0, 2, 3, 1)[0, ...].cpu().numpy().reshape(-1, 1)[::ds]
            tpts = self.mockup_opts.cpu().numpy()[:, ::ds]
            timg = self.target_gray.permute(0, 2, 3, 1)[0, ...].cpu().numpy().reshape(-1, 1)[::ds]
            fig = mlab.figure(bgcolor=(.5, .5, .5))
            mlab_rgbd(gpts, colors=gimg, size=.05, show_opt=False, fig=fig)
            mlab_rgbd(tpts, colors=timg, size=.05, show_opt=True, fig=fig)

    def plot_img_comparison(self):

            if isinstance(self.rimg, torch.Tensor):
                self.rimg = self.rimg.detach().cpu().numpy()

            if isinstance(self.nimg, torch.Tensor):
                self.nimg = self.nimg.detach().cpu().numpy()

            _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            axs[0].imshow(self.rimg)
            axs[0].set_title('before transformation')
            axs[1].imshow(self.nimg)
            axs[1].set_title('after transformation')
            plt.show()

    def test_all(self):

        self.test_point_fusion()


if __name__ == '__main__':
    unittest.main()
