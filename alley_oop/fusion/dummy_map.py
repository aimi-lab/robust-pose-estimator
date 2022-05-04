import torch
from alley_oop.pose.frame_class import FrameClass
from alley_oop.fusion.surfel_map import SurfelMap


class DummyMap(SurfelMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_frame = None

    def fuse(self, dept: torch.Tensor, gray: torch.Tensor, normals: torch.Tensor = None, pmat: torch.Tensor = None):
        super().fuse(dept, gray, normals, pmat)
        self.last_frame = FrameClass(gray, dept, normals=normals.view(1,3,*self.img_shape))

    def render(self, intrinsics: torch.tensor=None, extrinsics: torch.tensor=None):
        return self.last_frame

    def transform_cpy(self, transform):
        assert transform.shape == (4, 4)
        opts = transform[:3,:3]@self.opts + transform[:3,3,None]
        normals = transform[:3,:3] @ self.nrml
        return DummyMap(opts=opts, normals=normals, kmat=self.kmat, gray=self.gray, img_shape=self.img_shape,
                         radi=self.radi).to(self.device)
