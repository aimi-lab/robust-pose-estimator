import numpy as np
import imageio
import pickle
from skimage import transform, util
from mayavi import mlab

from utils.paths import SCARED_ROOT_PATH, get_scared_abspath
from utils.pinhole import reverse_project, forward_project
from utils.mlab_plot import mlab_rgbd, mlab_plot
from utils.pfm_handler import load_pfm
from pose.feat_pose_estimation import FeatPoseEstimator

def truncate_quantile(arr, p=0.01):

    lo_thresh = np.quantile(dis0, p)
    hi_thresh = np.quantile(dis0, 1-p)
    arr[arr<lo_thresh] = lo_thresh
    arr[arr>hi_thresh] = hi_thresh

    return arr

depth_path = SCARED_ROOT_PATH.parent / 'generated_depth_log_1641508997'
depth_list = sorted(depth_path.rglob('*d_*.pfm'))

calib_list = []
fnimg_list = []
for k_idx in range(1, 4):
    ipair_path = get_scared_abspath(d_idx=1, k_idx=1)
    ipair_list = sorted((ipair_path / 'data' / 'video_frames').rglob('*.*'))
    calib_list += ipair_list[0::3]
    fnimg_list += ipair_list[1::3]

feats_list = sorted((ipair_path / 'data' / 'superglue_results').rglob('*.npz'))
fname_pair = str(feats_list[0].name).split('_')[:2]
frame_jump = int(fname_pair[1][:-1]) - int(fname_pair[0][:-1])

# skip data which doesn't exist
depth_list = depth_list[:len(feats_list)]
fnimg_list = fnimg_list[:len(feats_list)]
calib_list = calib_list[:len(feats_list)]

assert len(calib_list) == len(fnimg_list) == len(depth_list) == len(feats_list), 'unequal number of image, disparity and calibration files'

# load data
i = 0
j = i+frame_jump
dis0, _ = load_pfm(depth_list[i])
dis1, _ = load_pfm(depth_list[j])
img0 = imageio.imread(fnimg_list[i])
img1 = imageio.imread(fnimg_list[j])

with open(calib_list[i],'rb') as f: cal0 = pickle.load(f)
with open(calib_list[j],'rb') as f: cal1 = pickle.load(f)

fnpz = np.load(feats_list[i])

# prepare data
resolution = (256, 512)
dis0 = truncate_quantile(dis0)
dis1 = truncate_quantile(dis1)
dis0 = transform.resize(dis0, resolution)
dis1 = transform.resize(dis1, resolution)
img0 = util.img_as_ubyte(transform.resize(img0, resolution))
img1 = util.img_as_ubyte(transform.resize(img1, resolution))

# feature matches
kpt0 = fnpz[fnpz.files[0]]
kpt1 = fnpz[fnpz.files[1]]
midx = fnpz[fnpz.files[2]]
conf = fnpz[fnpz.files[3]]
feat = [kpt0[midx>-1].T.astype(np.uint16), kpt1[midx][midx>-1].T.astype(np.uint16), conf[midx>-1]]
fpt0 = np.vstack([feat[0], np.ones(len(feat[0].T))])
fpt1 = np.vstack([feat[1], np.ones(len(feat[1].T))])

# image coordinates
x_coords = np.arange(0, resolution[1])
y_coords = np.arange(0, resolution[0])
x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten(), np.ones(len(x_mesh.flatten()))])

# 2D to 3D projection
bas0 = abs(cal0['T'][0][0])
bas1 = abs(cal1['T'][0][0])
emat = np.hstack([np.eye(3), np.array([[0, 0, 0]]).T])
opt0 = reverse_project(ipts, cal0['M1'], emat, dis0.flatten(), base=bas0)
opt1 = reverse_project(ipts, cal1['M1'], emat, dis1.flatten(), base=bas1)
rpts = reverse_project(fpt0, cal0['M1'], emat, dis0[feat[0][1], feat[0][0]].flatten(), base=bas0)
qpts = reverse_project(fpt1, cal1['M1'], emat, dis1[feat[1][1], feat[1][0]].flatten(), base=bas1)

# pose estimation
pose = FeatPoseEstimator(rpts, qpts, confidence=feat[-1])
pose.estimate(dim_weights=[.3, .3, .3])
tvec = pose.tvec
rmat = pose.rmat
print(tvec, rmat)

# map query points
mpts = rmat @ qpts + tvec
ppts = forward_project(np.vstack([mpts, np.ones(mpts.shape[1])]), cal0['M1'])
wpts = forward_project(np.vstack([qpts, np.ones(qpts.shape[1])]), cal0['M1'])

# 3D plot
fig = mlab.figure(bgcolor=(.5, .5, .5))
#mlab_rgbd(opt0, colors=img0.reshape(-1, 3), size=0.1, show_opt=False, fig=fig)
#mlab_rgbd(opt1, colors=img1.reshape(-1, 3), show_opt=False, fig=fig)
mlab_plot(rpts, colors=np.repeat([(0.5, 0, 0)], repeats=len(rpts.T), axis=0), size=1, show_opt=False, fig=fig)
mlab_plot(qpts, colors=np.repeat([(1.0, 0, 0)], repeats=len(rpts.T), axis=0), size=1, show_opt=False, fig=fig)
mlab_plot(mpts, colors=np.repeat([(0.0, 0, 0)], repeats=len(rpts.T), axis=0), size=1, show_opt=False, fig=fig)
mlab.show()

# 2D plot
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img0)
axs[0, 1].imshow(img1)
axs[0, 0].plot(feat[0][0], feat[0][1], 'g.')
axs[0, 1].plot(feat[1][0], feat[1][1], 'b.')
axs[0, 0].plot(ppts[0], ppts[1], 'bx')
axs[0, 0].plot(wpts[0], wpts[1], 'rx')
axs[1, 0].imshow(dis0, cmap='gray')
axs[1, 1].imshow(dis1, cmap='gray')
plt.show()
