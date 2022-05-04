import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Viewer3D(object):
    def __init__(self, image_shape, blocking=False):
        super().__init__()
        self.blocking = blocking
        self.exit_loop = not blocking
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.viewer = o3d.visualization.VisualizerWithKeyCallback()
        self.viewer.register_key_callback(81, self.exit_loop_callback)
        self.viewer.create_window(width=self.image_width,
                             height=self.image_height, visible=True)

        # get mesh
        self.control = self.viewer.get_view_control()
        self.ref_view = self.control.convert_to_pinhole_camera_parameters()
        opt = self.viewer.get_render_option()
        opt.background_color = np.asarray([55 / 255.0, 55 / 255.0, 55 / 255.0])
        self.pcd = None
        plt.ion()
        plt.show()

    def pose2view(self, pose):
        self.ref_view.extrinsic = np.linalg.pinv(pose)
        return self.ref_view

    def exit_loop_callback(self, dummy):
        self.exit_loop = True

    def __call__(self, pose, pcd=None, add_pcd=None, zoom=0.5, frame=None, synth_frame=None):
        # define viewer
        if frame is not None:
            if synth_frame is not None:
                img = frame.to_numpy()[1]
                img_synth = synth_frame.to_numpy()[1]
                img_view = np.concatenate((img, img_synth), axis=1)
                plt.imshow(img_view)
                plt.axis('off')
                plt.draw()
                plt.pause(0.0001)
        self.exit_loop = not self.blocking
        if self.blocking:
            print('blocking mode: press q to continue')
        if pcd is not None:
            self.viewer.remove_geometry(self.pcd, reset_bounding_box=True)
            self.pcd = pcd
            self.viewer.add_geometry(self.pcd)
        if add_pcd is not None:
            self.viewer.add_geometry(add_pcd)
        self.control.convert_from_pinhole_camera_parameters(self.pose2view(pose))
        self.control.set_zoom(zoom)
        self.viewer.poll_events()
        self.viewer.update_renderer()
        while not self.exit_loop:
            self.viewer.poll_events()
            self.viewer.update_renderer()
        if add_pcd is not None:
            self.viewer.remove_geometry(add_pcd)
