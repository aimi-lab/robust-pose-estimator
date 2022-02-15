#!.venv/bin/python

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import imageio

import os, sys
sys.path.insert(0, os.getcwd()) 

from alley_oop.utils.paths import SEGMEN_ROOT_PATH
from alley_oop.utils.pfm_handler import load_pfm, save_pfm
from alley_oop.geometry.pinhole_transforms import reverse_project, create_img_coords_np
from alley_oop.geometry.quaternions import quat2rmat
from alley_oop.interpol.surf_mappings import surf_interpol
from alley_oop.utils.exr_handler import load_exr, exr2gry

PLOT_OPT = True
SAVE_OPT = False

if __name__ == '__main__':

    # path handling
    data_dir = Path(SEGMEN_ROOT_PATH) / 'porcine_video' / '20180731_porcine_kidney_part0019'
    part_dir = str(data_dir.name).split('_')[-1]
    plys_dir = data_dir / (part_dir + '_s15_plys')
    outp_dir = data_dir / 'disparity_residuals_10.0fps'
    outp_dir.mkdir(exist_ok=True)

    # get camera pose from freiburg file
    fnames = sorted(data_dir.rglob('*.freiburg'))
    with open(fnames[0], 'r') as f: freiburg_list = [line for line in f]

    # get left disparity maps w/o tools
    dnames = sorted((data_dir / 'disparity_frames_10.0fps').rglob('*.pfm'))
    dnames = sorted((Path('/home/chris/Documents') / ('rgbd_'+part_dir+'_s15') / 'dept').rglob('*.exr'))

    # get pointcloud from ply file
    pnames = sorted(plys_dir.rglob('*.ply'), key=lambda s: int(str(s.name).split('.')[1][3:]))

    # intrinsics
    kmat = np.diag([525.8345947265625, 525.7257690429688, 1])
    kmat[0, -1] = 320
    kmat[1, -1] = 240

    # move future point clouds to present in an attempt to have the current depth represented inside
    gap = 10
    freiburg_list = freiburg_list[:-gap]
    dnames = dnames[:-gap]
    pnames = pnames[gap:]

    assert len(freiburg_list) == len(dnames[1:]) == len(pnames), 'length mismatch'

    for i, (pline, dname, pname) in enumerate(zip(freiburg_list, dnames[1:], pnames)):
        
        # load pose
        pose = np.array([float(el) for el in pline.strip('\n').split(' ')])
        tvec = pose[1:4][..., None]
        rmat = quat2rmat(pose[-4:])

        # load pointcloud
        plyd = PlyData.read(pname)
        verx = plyd['vertex']
        pcld = np.array([verx[t] for t in ('x', 'y', 'z')])

        if pcld.shape[1] > 3:
            
            # create image coordinates
            ipts = create_img_coords_np(y=kmat[1, -1]*2, x=kmat[0, -1]*2)   
            if str(dname).lower().endswith('pfm'):
                # load disparity
                disp = load_pfm(dname)[0]
                disp = disp[::2, ::2] / 2
                rgap = (512-480) // 2
                disp = disp[rgap:-rgap, :]
                # forward project
                opts = reverse_project(ipts, kmat, rmat, tvec, disp=disp, base=4.3590497970581055)
                opts = opts / 1000 # mm to meters
                opts = opts * 15
            elif str(dname).lower().endswith('exr'):
                fexr = load_exr(str(dname))
                dept = exr2gry(fexr)
                opts = reverse_project(ipts, kmat, rmat, tvec, dept=dept)

            # compute residuals in z dimension (point-wise comparison too expensive)
            residuals = surf_interpol(pcld, opts, method='bilinear', fill_val=float('NaN'))

            # convert residuals to writable uint8 image
            residuals[np.isnan(residuals)] = np.nanmax(residuals)
            residuals = residuals.reshape(int(kmat[1, -1]*2), int(kmat[0, -1]*2))
        else:
            residuals = np.zeros([int(kmat[1, -1]*2), int(kmat[0, -1]*2)], dtype=np.uint8)

        if SAVE_OPT:
            # save folating point residuals in 2-D map
            fname = str(outp_dir / (dname.name.split('.')[0]+'.pfm'))
            save_pfm(residuals, fname)

            # save residual depth as 2-D image
            resid_img = np.round(residuals/np.max(residuals)*255, 0).astype(np.uint8) if np.max(residuals) > 0 else residuals.astype(np.uint8)
            imageio.imwrite(fname.replace('.pfm', '.png'), resid_img)

        # plots for debug purposes
        if PLOT_OPT and residuals.sum() > 0 and i == 4*50:

            # plot ideally overlapping point clouds
            from alley_oop.utils.mlab_plot import mlab_plot
            from mayavi import mlab
            fig = mlab.figure(bgcolor=(1, 1, 1))
            mlab_plot(pcld, size=.01, fig=fig, colors=.25*np.ones(pcld.shape[1]), show_opt=False)
            mlab_plot(opts, size=.01, fig=fig, colors=.75*np.ones(opts.shape[1]))

            # plot residuals map
            plt.figure()
            plt.imshow(residuals)
            plt.show()
        
        # percentage
        percentage = round(i/len(pnames)*100, 1)
        print('\r '+str(dir) + str(percentage)+'%', end='', flush=True)
