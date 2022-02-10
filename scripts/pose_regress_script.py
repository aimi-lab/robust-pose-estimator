import numpy as np
import matplotlib.pyplot as plt
import imageio
import pickle
from skimage import transform, util
from mayavi import mlab
from pathlib import Path
import json
from tifffile import tifffile

from alley_oop.utils.paths import SCARED_ROOT_PATH, get_scared_abspath
from alley_oop.pinhole.pinhole_transforms import reverse_project, forward_project, create_img_coords
from alley_oop.utils.mlab_plot import mlab_rgbd, mlab_plot
from alley_oop.utils.pfm_handler import load_pfm
from alley_oop.utils.normals import normals_from_pca, get_ray_surfnorm_angle
from alley_oop.metrics.projected_photo_loss import dual_projected_photo_loss
from alley_oop.pose.feat_pose_estimation import FeatPoseEstimator


def clip_quantile(arr, p=1e-3):

    lo_thresh = np.quantile(dis0, p)
    hi_thresh = np.quantile(dis0, 1-p)
    arr[arr<lo_thresh] = lo_thresh
    arr[arr>hi_thresh] = hi_thresh

    return arr


def rescale_intrinsics(kmat, origin_size=None, target_size=None):

    kmat[0, ...] *= (target_size[1]/origin_size[1])
    kmat[1, ...] *= (target_size[0]/origin_size[0])

    return kmat


depth_path = SCARED_ROOT_PATH.parent / 'generated_depth_log_1642501697'
depth_list = sorted(depth_path.rglob('*d_*.pfm'))

calib_list = []
fnimg_list = []
for k_idx in range(1, 4):
    ipair_path = get_scared_abspath(d_idx=1, k_idx=1)
    ipair_list = sorted((ipair_path / 'data' / 'video_frames').rglob('*.*'))
    calib_list += ipair_list[0::3]
    fnimg_list += ipair_list[1::3]

feats_path = ipair_path / 'data' / 'superglue_results_const_gap_640x512_framegap1'
feats_list = sorted((feats_path).rglob('*.npz'))
fname_pair = str(feats_list[0].name).split('_')[:2]
frame_jump = int(fname_pair[1][:-1]) - int(fname_pair[0][:-1])
gtpos_list = sorted((ipair_path / 'data' / 'frame_data').rglob('*.json'))
scene_list = sorted((ipair_path / 'data' / 'scene_points').rglob('*.tiff'))

# skip data which doesn't exist
depth_list = depth_list[:len(feats_list)]
fnimg_list = fnimg_list[:len(feats_list)]
calib_list = calib_list[:len(feats_list)]

assert len(calib_list) == len(fnimg_list) == len(depth_list) == len(feats_list), 'unequal number of image, disparity and calibration files'

# plot & save settings
save_opt = 1
save_map = 0
plot_opt = 0
if plot_opt == 1: fig = mlab.figure(bgcolor=(.5, .5, .5))

# var init
stats = []
j = 0
pose_list = []
for i in range(0, len(feats_list)-frame_jump, frame_jump):

    # load data
    i = i if str(feats_path).__contains__('const_gap') else 0
    j = i+frame_jump if str(feats_path).__contains__('const_gap') else j+frame_jump
    dis0 = load_pfm(depth_list[i])[0]
    dis1 = load_pfm(depth_list[j])[0]
    img0 = imageio.imread(fnimg_list[i])
    img1 = imageio.imread(fnimg_list[j])
    fnpz = np.load(feats_list[i])
    pcl0 = tifffile.imread(scene_list[i])[:1024, :]
    pcl1 = tifffile.imread(scene_list[j])[:1024, :]
    with open(calib_list[i],'rb') as f: cal0 = pickle.load(f)
    with open(calib_list[j],'rb') as f: cal1 = pickle.load(f)
    with open(gtpos_list[i],'rb') as f: pos0 = np.array(json.load(f)['camera-pose'])
    with open(gtpos_list[j],'rb') as f: pos1 = np.array(json.load(f)['camera-pose'])
    if len(pose_list) == 0:
        pose_list.append([pos0[:3, :], pos0[:3, :]])

    # prepare data
    us = .5
    ds = int(2/us)
    resolution = (512*us, 640*us)
    cal0['M1'] = rescale_intrinsics(cal0['M1'], origin_size=img0.shape[:2], target_size=resolution)
    cal1['M1'] = rescale_intrinsics(cal1['M1'], origin_size=img1.shape[:2], target_size=resolution)
    dis0 = clip_quantile(dis0)
    dis1 = clip_quantile(dis1)
    dis0 = transform.resize(dis0, resolution)/(dis0.shape[1]/resolution[1])
    dis1 = transform.resize(dis1, resolution)/(dis1.shape[1]/resolution[1])
    img0 = util.img_as_ubyte(transform.resize(img0, resolution))
    img1 = util.img_as_ubyte(transform.resize(img1, resolution))
    pcl0 = pcl0[::ds, ::ds, :]
    pcl1 = pcl1[::ds, ::ds, :]

    # feature matches
    kpt0 = fnpz[fnpz.files[0]]
    kpt1 = fnpz[fnpz.files[1]]
    midx = fnpz[fnpz.files[2]]
    conf = fnpz[fnpz.files[3]]
    feat = [kpt0[midx>-1].T.astype(np.uint16), kpt1[midx][midx>-1].T.astype(np.uint16), conf[midx>-1]]
    feat[0] = (feat[0] * us).astype(np.int64)
    feat[1] = (feat[1] * us).astype(np.int64)
    fpt0 = np.vstack([feat[0], np.ones(len(feat[0].T))])
    fpt1 = np.vstack([feat[1], np.ones(len(feat[1].T))])
    fcl0 = pcl0[feat[0][1], feat[0][0], :].reshape(-1, 3).T
    fcl1 = pcl1[feat[1][1], feat[1][0], :].reshape(-1, 3).T
    fcl0 = fcl0[:, ~np.isnan(fcl0.sum(0))]
    fcl1 = fcl1[:, ~np.isnan(fcl1.sum(0))]
    conf = feat[-1]**.5

    # image coordinates
    ipts = create_img_coords(*resolution)

    # 2D to 3D projectionus
    bas0 = abs(cal0['T'][0][0])
    bas1 = abs(cal1['T'][0][0])
    opt0 = reverse_project(ipts, cal0['M1'], disp=dis0.flatten(), base=bas0)
    opt1 = reverse_project(ipts, cal1['M1'], disp=dis1.flatten(), base=bas1)
    rpts = reverse_project(fpt0, cal0['M1'], disp=dis0[feat[0][1], feat[0][0]].flatten(), base=bas0)
    qpts = reverse_project(fpt1, cal1['M1'], disp=dis1[feat[1][1], feat[1][0]].flatten(), base=bas1)

    # downsampling point cloud for topological fit
    divs = (64, 32)
    idcs = (ipts[0, ::200] >= resolution[1]//divs[1]) & \
            (ipts[0, ::200] <= resolution[1]//divs[1]*(divs[1]-1)) & \
            (ipts[1, ::200] >= resolution[0]//divs[0]) & \
            (ipts[1, ::200] <= resolution[0]//divs[0]*(divs[0]-1))
    tpt0 = np.vstack([opt0, np.mean(img0.reshape(-1, 3).T, axis=0)])[:, ::200]
    tpt1 = np.vstack([opt1, np.mean(img1.reshape(-1, 3).T, axis=0)])[:, ::200][:, idcs]

    # surface normal computation to exclude points facing away from camera
    naxs = normals_from_pca(rpts, distance=10, leafsize=10)
    angs = get_ray_surfnorm_angle(rpts, naxs)
    angs = np.abs(angs*180/np.pi)
    conf[angs>55] = 0
    if plot_opt == 3:
        from tests.unit_test_normals import NormalsTester
        NormalsTester.plot_normals(naxs, rpts)
        plt.show()

    # pose estimation
    #rpts, qpts = fcl0, fcl1
    estp = FeatPoseEstimator(rpts, qpts, confidence=conf)
    #estp = TopoPoseEstimator(tpt0, tpt1)
    #estp.estimate()
    estp.random_sample_consesus()
    tvec = estp.tvec
    rvec = estp.rvec
    rmat = estp.rmat
    wdim = estp.feat_wdims
    opts = rmat @ opt0 + tvec

    # evaluate stats
    feat_loss = estp.p_loss/len(estp.residual_fun(estp.p_star))
    foto_loss = dual_projected_photo_loss(img0, img1, dis0/bas0, dis1/bas1, rmat, tvec, cal0['M1'])
    if str(feats_path).__contains__('const_gap'):
        tvec_loss = np.mean((pos1[:3, -1] - pose_list[-1][1][:, -1]+tvec)**2)**.5
        rmat_loss = np.mean((pos1[:3, :3] - rmat)**2)**.5   #pose_list[-1][1][:, :-1]+
    else:
        tvec_loss = np.mean((pos1[:3, -1] - tvec)**2)**.5
        rmat_loss = np.mean((pos1[:3, :3] - rmat)**2)**.5
    print('Frame pair:              %s, %s' % (i, j))
    print('Translation (Eucl.):     %s (%s)' % (tvec.ravel(), sum(tvec.ravel()**2)**.5))
    print('Rotation:                %s' % rvec.ravel())
    print('Dim-weights:             %s' % wdim.ravel())
    print('Loss per feature:        %s' % str(feat_loss))
    print('Photometric loss:        %s' % str(foto_loss))
    print('Translation loss:        %s' % str(tvec_loss))
    print('Rotation loss:           %s' % str(rmat_loss))
    print('Iterations:              %s' % len(estp.p_list))
    print('\n')

    # store stats
    stats.append([estp.p_loss/len(estp.residual_fun(estp.p_star)), len(estp.p_list), foto_loss])
    pose_list.append([pos1[:3, :], np.hstack([rmat, tvec])])

    # write intermediate results to drive
    if save_opt:
        # write each pose as 4x4 matrix
        pose_fname = Path('.').parent / 'tests' / 'test_data' / str(feats_list[i].name).replace('matches.npz', 'pose_es.json')
        if not pose_fname.parent.exists(): pose_fname.parent.mkdir()
        pose_esmat = {'camera-pose': np.vstack([np.array(pose_list[-1])[1], np.array([0, 0, 0, 1])]).tolist()}
        pose_gtmat = {'camera-pose': np.vstack([np.array(pose_list[-1])[0], np.array([0, 0, 0, 1])]).tolist()}
        with open(str(pose_fname), 'w') as f: json.dump(pose_esmat, f, indent=4)
        #with open(str(pose_fname).replace('pose_es', 'pose_km'), 'w') as f: json.dump(pose_gtmat, f, indent=4)

    # write rgbd point clouds
    if save_map:
        rgbd_fname = Path('.').parent / 'tests' / 'test_data' / str(feats_list[i].name).replace('matches', 'rgbd')
        rgbd0 = np.dstack([opt0.T.reshape(*resolution, 3), img0])
        rgbd1 = np.dstack([opt1.T.reshape(*resolution, 3), img1])
        np.savez_compressed(rgbd_fname, rgbd0=rgbd0, rgbd1=rgbd1)

    # 3D plot
    if plot_opt == 1 and feat_loss:
        ds = 10
        mlab_rgbd(opt1[:, ::ds], colors=img0.reshape(-1, 3)[::ds], size=.1, show_opt=False, fig=fig)
        mlab_rgbd(opts[:, ::ds], colors=img1.reshape(-1, 3)[::ds], size=.1, show_opt=False, fig=fig)
    if plot_opt == 2:
        mpts = rmat @ rpts + tvec
        fig = mlab.figure(bgcolor=(.5, .5, .5))
        mlab_plot(rpts, colors=np.repeat([(1.0, 0, 0)], repeats=len(rpts.T), axis=0), size=1, show_opt=False, fig=fig)
        mlab_plot(qpts, colors=np.repeat([(0.5, 0, 0)], repeats=len(rpts.T), axis=0), size=1, show_opt=False, fig=fig)
        mlab_plot(mpts, colors=np.repeat([(0.0, 0, 0)], repeats=len(rpts.T), axis=0), size=1, show_opt=False, fig=fig)
        mlab.show()
    if plot_opt == 3:
        fcl0 = rmat @ fcl0 + tvec
        ds = 500
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(fcl0[0, ::ds], fcl0[1, ::ds], fcl0[2, ::ds], 's', color='b', markersize=3, label='reference points moved')
        ax.plot(fcl1[0, ::ds], fcl1[1, ::ds], fcl1[2, ::ds], 'o', color='g', markersize=3, label='current point cloud')
        plt.legend()
        plt.show()

    # 2D plot
    if plot_opt == 2:
        ppts = forward_project(rpts, cal0['M1'], rmat, tvec)
        from matplotlib import rc
        rc('text', usetex=True)
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        axs[0, 0].imshow(img0)
        axs[0, 1].imshow(img1)
        axs[0, 0].plot(feat[0][0], feat[0][1], 'g.', label='$\mathbf{x}^{(i)}_k$')
        axs[0, 1].plot(feat[1][0], feat[1][1], 'g.', label='$\mathbf{x}^{(i)}_l$')
        axs[0, 1].plot(ppts[0], ppts[1], 'bx', label='$\mathbf{\widetilde{x}}^{(i)}_l$')
        axs[1, 0].imshow(dis0, cmap='gray')
        axs[1, 1].imshow(dis1, cmap='gray')
        axs[0, 0].legend(loc="upper right")
        axs[0, 1].legend(loc="upper right")
        axs[0, 0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        axs[0, 1].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.tight_layout()
        #plt.savefig('feature_projected_loss.svg')
        plt.show()

avg_feat_loss, avg_iter, avg_foto_loss = np.mean(stats, axis=0).tolist()
print('avg_feat_loss:   %s' % avg_feat_loss)
print('avg_foto_loss:   %s' % avg_foto_loss)
print('avg_iter:        %s' % avg_iter)

if plot_opt == 1: mlab.show()