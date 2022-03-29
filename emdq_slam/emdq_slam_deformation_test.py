from alley_oop.geometry.camera import PinholeCamera
from alley_oop.metrics.point_cloud_metrics import pcl_ae, nearest_neighbour_dist
from emdq_slam.emdq_slam_pipeline import EmdqSLAM, EmdqGlueSLAM
from alley_oop.metrics.trajectory_metrics import absolute_trajectory_error
import cv2
import numpy as np
from alley_oop.phantom.deformable_texture_phantom import DeformableTexturePhantom
from viewer.slam_viewer import Viewer3d


def main(config):
    # load test data to create phantom
    img = cv2.cvtColor(cv2.resize(cv2.imread('../tests/test_data/000000l.png'), (640, 480)), cv2.COLOR_BGR2RGB)
    disparity = cv2.resize(cv2.imread('../tests/test_data/000000l.pfm', cv2.IMREAD_UNCHANGED), (640, 480)) / 2
    depth = 2144.878173828125 / disparity
    camera = PinholeCamera(np.array([[517.654052734375, 0, 298.4775085449219],
                                     [0, 517.5438232421875, 244.20501708984375],
                                     [0, 0, 1]]))
    phantom = DeformableTexturePhantom(img, depth, camera, n_deformation_nodes=1, steps=10)

    phantom.deform(deformation_param=3.0)
    #phantom.animate()


    disp_img = np.zeros((480 * 2, 640 * 2, 3), dtype=np.uint8)

    # for i, (img, depth, points3d) in enumerate(phantom):
    #     if i == 0:
    #         disp_img[:480, :640] = img
    #         disp_img[480:, :640] = depth[..., None]
    #     disp_img[:480, 640:] = img
    #     disp_img[480:, 640:] = depth[..., None]
    #     cv2.imshow('phantom', disp_img)
    #     cv2.waitKey(1)

    slam = EmdqGlueSLAM(camera, config['slam'])
    viewer = Viewer3d(blocking=True)
    trajectory = []
    for i, (img, depth, points3d) in enumerate(phantom):
        pose, inliers = slam(img, depth)
        if inliers == 0:
            break
        trajectory.append(pose)
        absolute_pcl_error = pcl_ae(points3d, slam.warp_canonical_model(current_reference=False))
        print(absolute_pcl_error)
        viewer(points3d[::20], slam.warp_canonical_model(current_reference=False), img.reshape(-1,3)[::20], nearest_neighbour_dist(points3d, slam.warp_canonical_model(current_reference=False)))

    ate_pos, ate_rot = absolute_trajectory_error(len(trajectory)*[np.eye(4)], trajectory)
    print(ate_pos)
    print(ate_rot)

if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='script to test EMDQ SLAM on a deformable phantom')

    parser.add_argument(
        '--config',
        type=str,
        default='configuration/default.yaml',
        help='Configuration file.'
    )

    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    main(config)
