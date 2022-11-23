from mayavi import mlab
import numpy as np


def mlab_plot(arr, colors=None, size=10, show_opt=True, save_opt=False, skip_opt=False, fig=None):

    if skip_opt:
        return True

    # use z as default color
    colors = arr[2] if colors is None else colors

    fig = mlab.figure(bgcolor=(1, 1, 1)) if fig is None else fig
    pts = mlab.points3d(arr[0], arr[1], arr[2], 
                                        #colors,
                                        mode="sphere",
                                        #colormap='spectral',# 'bone', 'copper', 'gnuplot'
                                        #color=(0, 1, 0),# Used a fixed (r,g,b) instead
                                        scale_factor=size,
                                        figure=fig,
                                        )
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.mlab_source.dataset.point_data.scalars = colors
    mlab.show() if show_opt else None
    mlab.savefig('./figure.ps') if save_opt else None


def mlab_rgbd(arr, colors=None, size=10, show_opt=True, save_opt=False, skip_opt=False, fig=None):

    if skip_opt:
        return True

    # use z as default color
    ptslut = (np.ones((len(arr[0]), 4))*255).astype(np.uint8)
    ptslut[..., :3] = np.repeat(arr[2][np.newaxis], 3, axis=0).T if colors is None else colors.astype(np.uint8)

    fig = mlab.figure(bgcolor=(1, 1, 1)) if fig is None else fig
    pts = mlab.quiver3d(arr[0], arr[1], arr[2],
                                        np.ones(len(arr[0])), np.ones(len(arr[0])), np.ones(len(arr[0])),
                                        scalars=np.arange(len(arr[0])), # indices for colors when setting "color by scalar"
                                        scale_factor=size,
                                        mode="sphere",
                                        figure=fig,
                                        )
    pts.glyph.color_mode = 'color_by_scalar' # Color by scalar
    pts.module_manager.scalar_lut_manager.lut.table = ptslut
    mlab.draw()
    mlab.show() if show_opt else None
    mlab.savefig('./figure.ps') if save_opt else None

def plot_intersections(pts_intersect, npts, plot_opt=True, show_opt=True, scale_factor=0.05):

    if not plot_opt:
        return False

    pts = mlab.points3d(np.hstack([pts_intersect[0], npts[0]]), np.hstack([pts_intersect[1], npts[1]]), np.hstack([pts_intersect[2], npts[2]]), scale_factor=scale_factor)

    ascend_idx = np.arange(len(npts[0]))
    connections = np.vstack([ascend_idx, ascend_idx+len(npts[0])])
    pts.mlab_source.dataset.lines = np.array(connections).T

    tube = mlab.pipeline.tube(pts, tube_radius=0.005)
    tube.filter.radius_factor = .1
    mlab.pipeline.surface(tube, color=(1, 0, 0))
    mlab.show() if show_opt else None

    return True
