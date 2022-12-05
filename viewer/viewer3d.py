try:
    import numpy as np
    import matplotlib.pyplot as plt
except:
    pass


class Viewer3D(object):
    def __init__(self, image_shape, blocking=False):
        super().__init__()
        import open3d as o3d
        self.blocking = blocking
        self.exit_loop = not blocking
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.viewer = o3d.visualization.VisualizerWithKeyCallback()
        self.viewer.register_key_callback(81, self.exit_loop_callback)
        self.viewer.register_key_callback(68, self.deform_callback)
        self.viewer.create_window(width=self.image_width,
                             height=self.image_height, visible=True)

        # get mesh
        self.control = self.viewer.get_view_control()
        self.ref_view = self.control.convert_to_pinhole_camera_parameters()
        opt = self.viewer.get_render_option()
        opt.background_color = np.asarray([55 / 255.0, 55 / 255.0, 55 / 255.0])
        self.pcd = None
        self.def_pcd = None
        plt.ion()
        plt.show()
        self.is_deformed = False

    def pose2view(self, pose):
        self.ref_view.extrinsic = pose.inv().matrix.numpy()
        return self.ref_view

    def exit_loop_callback(self, dummy):
        self.exit_loop = True

    def deform_callback(self, dummy):
        self.pose = self.control.convert_to_pinhole_camera_parameters()
        self.is_deformed = ~self.is_deformed
        if self.is_deformed:
            print("deformed/current")
        else:
            print("canonical/current")
        self.viewer.remove_geometry(self.pcd)
        def_pcd = self.def_pcd
        self.def_pcd = self.pcd
        self.pcd = def_pcd
        self.viewer.add_geometry(self.pcd)
        self.control.convert_from_pinhole_camera_parameters(self.pose)

    def __call__(self, pose, pcd=None, add_pcd=None, zoom=0.5, frame=None, synth_frame=None, def_pcd=None, idx:int=0):
        # plot input frame and synthesized frame
        self.pose = self.pose2view(pose)
        self.is_deformed = False
        self.def_pcd = def_pcd
        if (frame is not None):
            fig, ax = plt.subplots(2, 3, num=1, clear=True, figsize=(10,8))
            if (frame is not None) & (synth_frame is not None):
                img, _,_, depth, *_, conf = frame.to_numpy()
                ax[0,0].imshow(img)
                ax[0, 0].axis('off')
                ax[0, 0].set_title('I_t')
                ax[0, 1].imshow(depth, vmin=0, vmax=1)
                ax[0, 1].axis('off')
                ax[0, 1].set_title('depth_t')
                ax[0, 2].imshow(conf, vmin=0, vmax=1)
                ax[0, 2].axis('off')
                ax[0, 2].set_title('weights_t')
                img, _,_, depth, *_, conf = synth_frame.to_numpy()
                ax[1, 0].imshow(img)
                ax[1, 0].axis('off')
                ax[1, 0].set_title('I_t-1')
                ax[1, 1].imshow(depth, vmin=0, vmax=1)
                ax[1, 1].axis('off')
                ax[1, 1].set_title('depth_t-1')
                ax[1, 2].imshow(conf, vmin=0, vmax=1)
                ax[1, 2].axis('off')
                ax[1, 2].set_title('weights_t-1')
            plt.draw()
            plt.savefig('dummy.png')
            plt.pause(0.0001)

        self.exit_loop = not self.blocking
        if self.blocking:
            print('blocking mode: press q to continue, d to deform model')
            print('current depth with texture, scene error color coded.')

        if add_pcd is not None:

            self.viewer.add_geometry(add_pcd)
        if pcd is not None:
            self.viewer.remove_geometry(self.pcd, reset_bounding_box=True)
            self.pcd = pcd
            self.viewer.add_geometry(self.pcd)
        self.control.convert_from_pinhole_camera_parameters(self.pose2view(pose))
        #self.control.set_zoom(zoom)
        self.viewer.poll_events()
        self.viewer.update_renderer()
        while not self.exit_loop:
            self.viewer.poll_events()
            self.viewer.update_renderer()
        if add_pcd is not None:
            self.viewer.remove_geometry(add_pcd)

