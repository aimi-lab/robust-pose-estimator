import numpy as np
import imageio
import pickle
from skimage import transform, util
import json
from tifffile import tifffile

from alley_oop.utils.paths import SCARED_ROOT_PATH, get_scared_abspath
from alley_oop.geometry.pinhole_transforms import reverse_project, create_img_coords_np
from alley_oop.utils.pfm_handler import load_pfm


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

def load_tiff_pcl(d_idx:int=1, k_idx:int=1, fname=None):

    if fname is None: fname = get_scared_abspath(d_idx, k_idx, 'left_depth_map.tiff')

    # tiff-based depth
    tpcl = np.array(tifffile.imread(fname))

    return tpcl

depth_path = SCARED_ROOT_PATH.parent / 'generated_depth_log_1642502310'#'generated_depth_log_1642501697'#'generated_depth_log_1641508997'#'generated_depth_log_1642441959'#'generated_depth_log_1642442355'#
depth_list = sorted(depth_path.rglob('*d_*.pfm'))

calib_list = []
fnimg_list = []
ipair_path = get_scared_abspath(d_idx=1, k_idx=1)
ipair_list = sorted((ipair_path / 'data' / 'video_frames').rglob('*.*'))
scene_list = sorted((ipair_path / 'data' / 'scene_points').rglob('*.*'))
calib_list += ipair_list[0::3]
fnimg_list += ipair_list[1::3]

# load data
i = 0
dis0 = load_pfm(depth_list[i])[0]
img0 = imageio.imread(fnimg_list[i])
with open(calib_list[i],'rb') as f: cal0 = pickle.load(f)
pclg = load_tiff_pcl(fname=scene_list[i])[:1024, ...]

# prepare data
resolution = (1024, 1280)
us = .5
ds = int(2/us)
cal0['M1'] = rescale_intrinsics(cal0['M1'], origin_size=img0.shape[:2], target_size=resolution)
dis0 = clip_quantile(dis0)
dis0 = transform.resize(dis0, resolution)/(dis0.shape[1]/resolution[1])
img0 = util.img_as_ubyte(transform.resize(img0, resolution))
pclg = pclg.reshape(-1, 3).T

# remove camera center
gtpos_list = sorted((ipair_path / 'data' / 'frame_data').rglob('*.json'))
with open(gtpos_list[i],'rb') as f: pose = np.array(json.load(f)['camera-pose'])
tvec = pose[:3, -1][np.newaxis].T
rmat = pose[:3, :3]
orgn = np.zeros([3, 1])

# image coordinates
ipts = create_img_coords_np(*resolution)

# 2D to 3D projection
bas0 = abs(cal0['T'][0][0])
opts = reverse_project(ipts, cal0['M1'], disp=dis0.flatten(), base=bas0)

# select points where ground-truth reference is available (exclude NaNs)
vidx = (~np.isnan(pclg.sum(0))) & (pclg.sum(0) != 0)
pcln = pclg[:, vidx]
epts = opts[:, vidx]

# coordinate reference transformation
#pclg = np.linalg.pinv(rmat) @ pclg - tvec
#opts = rmat @ opts + tvec

# numerical displacement errors
abs_dev_x = np.mean((epts[0] - pcln[0])**2)**.5
abs_dev_y = np.mean((epts[1] - pcln[1])**2)**.5
abs_dev_z = np.mean((epts[2] - pcln[2])**2)**.5
rel_dev_x = 100*abs_dev_x/(pcln[0].max()-pcln[0].min())
rel_dev_y = 100*abs_dev_x/(pcln[1].max()-pcln[1].min())
rel_dev_z = 100*abs_dev_x/(pcln[2].max()-pcln[2].min())
print("abs. x-RMSE: %s" % str(abs_dev_x))
print("abs. y-RMSE: %s" % str(abs_dev_y))
print("abs. z-RMSE: %s" % str(abs_dev_z))
print("rel. x-RMSE: %s" % str(rel_dev_x))
print("rel. y-RMSE: %s" % str(rel_dev_y))
print("rel. z-RMSE: %s" % str(rel_dev_z))

# 3D plot
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca(projection='3d')
ds = 500
ax.plot(*orgn, marker='s', color='orange', markersize=6, linestyle='none', label='keyframe')
ax.plot(epts[0, ::ds], epts[1, ::ds], epts[2, ::ds], 's', color='b', markersize=3, label='estimated points')
ax.plot(pcln[0, ::ds], pcln[1, ::ds], pcln[2, ::ds], 'o', color='g', markersize=3, label='ground-truth points')
plt.legend(loc='upper right')
plt.show()
