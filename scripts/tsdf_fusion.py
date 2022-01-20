# http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html

import numpy as np
import open3d as o3d
from tifffile import tifffile
import imageio
import pickle

from scripts.trajectory_plots import load_scared_pose
from alley_oop.utils.paths import get_scared_abspath, SCARED_ROOT_PATH
from alley_oop.utils.pfm_handler import load_pfm

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

default_opt = True

depth_path = SCARED_ROOT_PATH.parent / 'generated_depth_log_1642502310'
depth_list = sorted(depth_path.rglob('*d_*.pfm'))
ipair_path = get_scared_abspath(d_idx=1, k_idx=1)
ipair_list = sorted((ipair_path / 'data' / 'video_frames').rglob('*.*'))
calib_list = ipair_list[0::3]
fnimg_list = ipair_list[1::3]
scene_list = sorted((ipair_path / 'data' / 'scene_points').rglob('*.tiff'))
camera_poses = read_trajectory("../other_repos/Open3D/examples/test_data/RGBD/odometry.log") if default_opt else load_scared_pose(meth='orbslam2_stereo_results')[:5]#
s = 1

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0 * s,
    sdf_trunc=0.04 * s,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

for i in range(len(camera_poses)):
    print("Integrate {:d}-th image into the volume.".format(i))
    img0 = imageio.imread(fnimg_list[i])
    pcl0 = (30*load_pfm(depth_list[i])[0]).astype(np.uint16)#np.round(25*tifffile.imread(scene_list[i])[:1024, :, -1]).astype(np.uint16)
    color = o3d.io.read_image("../other_repos/Open3D/examples/test_data/RGBD/color/{:05d}.jpg".format(i)) if default_opt else o3d.geometry.Image(img0)#
    depth = o3d.io.read_image("../other_repos/Open3D/examples/test_data/RGBD/depth/{:05d}.png".format(i)) if default_opt else o3d.geometry.Image(pcl0)#
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, 
            depth, 
            depth_trunc=np.array(depth).max(),
            convert_rgb_to_intensity=False
            )

    cal_obj=o3d.camera.PinholeCameraIntrinsic()
    with open(calib_list[i],'rb') as f: cal0 = pickle.load(f)
    cal_obj.set_intrinsics(
                    width=img0.shape[1], 
                    height=img0.shape[0],
                    fx=cal0['M1'][0, 0], 
                    fy=cal0['M1'][1, 1], 
                    cx=cal0['M1'][0, -1], 
                    cy=cal0['M1'][1, -1],
                    )

    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault) if default_opt else cal_obj,
        np.linalg.inv(camera_poses[i].pose) if default_opt else np.linalg.inv(camera_poses[i])
        )

# extract mesh
print("Extract a triangle mesh from the volume and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh],
                                  front=[0.5297, -0.1873, -0.8272],
                                  lookat=[2.0712, 2.0312, 1.7251],
                                  up=[-0.0558, -0.9809, 0.1864],
                                  zoom=0.47
                                  )