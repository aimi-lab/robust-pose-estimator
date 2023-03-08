import numpy as np
import cv2
import os
import torch

class ViewRenderer(object):
    def __init__(self, image_shape, outpath):
        super().__init__()
        import open3d as o3d
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.viewer = o3d.visualization.VisualizerWithKeyCallback()
        self.viewer.create_window(width=self.image_width,
                             height=self.image_height, visible=True)
        self.vid_writer = cv2.VideoWriter(os.path.join(outpath, 'vis.mp4'), cv2.VideoWriter_fourcc(*'MP4V'),
                                     25.0, (image_shape[1], image_shape[0]-1))
        # get mesh
        self.control = self.viewer.get_view_control()
        self.ref_view = self.control.convert_to_pinhole_camera_parameters()
        opt = self.viewer.get_render_option()
        opt.background_color = np.asarray([55 / 255.0, 55 / 255.0, 55 / 255.0])
        self.pcd = None
        self.def_pcd = None
        self.is_deformed = False

    def __del__(self):
        print("release writer")
        self.vid_writer.release()

    def pose2view(self, pose):
        self.ref_view.extrinsic = pose.inv().matrix.numpy()
        return self.ref_view

    def __call__(self, pose, pcd):
        # plot input frame and synthesized frame
        self.pose = self.pose2view(pose)

        self.viewer.remove_geometry(self.pcd, reset_bounding_box=True)
        self.pcd = pcd
        self.viewer.add_geometry(self.pcd)
        self.control.convert_from_pinhole_camera_parameters(self.pose2view(pose))
        self.control.set_zoom(0.5)
        self.viewer.poll_events()
        self.viewer.update_renderer()
        # self.viewer.run()
        image = self.viewer.capture_screen_float_buffer(False)
        self.vid_writer.write(cv2.cvtColor((255*np.asarray(image)).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return image

