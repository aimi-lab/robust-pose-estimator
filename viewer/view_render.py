import open3d as o3d
import numpy as np


class Render(object):
    def __init__(self, pcd, image_width=640, image_height=480, blocking=False):
        super().__init__()
        self.pcd = pcd
        self.blocking = blocking
        self.exit_loop = not blocking
        self.image_height = image_height
        self.image_width = image_width
        self.viewer = o3d.visualization.VisualizerWithKeyCallback()
        self.viewer.register_key_callback(81, self.exit_loop_callback)
        self.viewer.create_window(width=self.image_width,
                             height=self.image_height, visible=True)

        # get mesh
        self.viewer.add_geometry(self.pcd)
        self.control = self.viewer.get_view_control()
        self.ref_view = self.control.convert_to_pinhole_camera_parameters()
        opt = self.viewer.get_render_option()
        opt.background_color = np.asarray([55 / 255.0, 55 / 255.0, 55 / 255.0])

    def pose2view(self, pose):
        self.ref_view.extrinsic = np.linalg.pinv(pose)
        #ref_view.extrinsic = np.linalg.inv(pose)
        return self.ref_view

    def exit_loop_callback(self, dummy):
        self.exit_loop = True

    def render(self, pose, pcd=None, add_pcd=None, zoom=0.5):
        # define viewer
        self.exit_loop =  not self.blocking
        if pcd is not None:
            self.viewer.remove_geometry(self.pcd, reset_bounding_box=True)
            self.pcd = pcd
            self.viewer.add_geometry(self.pcd)
        if add_pcd is not None:
            self.viewer.add_geometry(add_pcd)
        self.control.convert_from_pinhole_camera_parameters(self.pose2view(pose))
        self.control.set_zoom(zoom)
        while not self.exit_loop:
            self.viewer.poll_events()
            self.viewer.update_renderer()
        image = self.viewer.capture_screen_float_buffer(False)
        if add_pcd is not None:
            self.viewer.remove_geometry(add_pcd)

        return image

# import sys
# sys.path.append('../')
# from utils.elastic_fusion_utils import load_trajectory, load_map
#
# trajectory = load_trajectory('/home/mhayoz/research/innosuisse_surgical_robot/01_Datasets/02_segmentation/intuitive_segmentation/porcine_video/20180731_porcine_kidney_part0019/elastic_fusion_c5_10.0fps_results/part0019.klg.freiburg')
# map_pcl, _ = load_map('/home/mhayoz/research/innosuisse_surgical_robot/01_Datasets/02_segmentation/intuitive_segmentation/porcine_video/20180731_porcine_kidney_part0019/elastic_fusion_c5_10.0fps_results/part0019.klg.ply')
# renderer = Render(map_pcl)
# img = renderer.render(trajectory[30])
# img2 = renderer.render(trajectory[500])
# import matplotlib.pyplot as plt
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(img2)
# plt.show()