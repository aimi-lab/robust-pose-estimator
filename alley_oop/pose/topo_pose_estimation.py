import numpy as np
from scipy import spatial
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator

from alley_oop.pose.feat_pose_estimation import FeatPoseEstimator


class TopoPoseEstimator(FeatPoseEstimator):
    """ pose regression based on topological geometry and non-linear least squares solver """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # override feature dimensions to account for ignored spatial dimensions (x, y)
        self.dimensions = int(self.feat_refer.shape[0]-2)
        self.feat_wdims = np.ones([self.dimensions])/self.dimensions

        # override confidence values in case of dimensionality mismatch
        if self.confidence is not None and self.confidence.shape[-1] != self.feat_refer.shape[-1]:
            self.confidence = None

        self.method = 'cubic'
        self.fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else 1000
        self.interpolators = self.create_interpolator()

    def create_interpolator(self):
        """ create interpolator (linear or cubic) from reference points """

        interpolators = []
        for i in range(self.dimensions):
            tri = spatial.Delaunay(self.feat_query[:2].copy().T)
            InterpolatorClass = CloughTocher2DInterpolator if self.method.lower() == 'cubic' else LinearNDInterpolator
            interpolators.append(InterpolatorClass(points=tri,
                                                   values=self.feat_query[i+2].copy(),
                                                   rescale=False,
                                                   fill_value=self.fill_value,
                                                   ))
            
        return interpolators

    def compute_diff(self, rpts, qpts=None):
        
        diff_arr = np.zeros([rpts.shape[0]-2, rpts.shape[1]])
        for i, interpolator in enumerate(self.interpolators):
            diff_arr[i] = interpolator(rpts[0], rpts[1]) - rpts[i+2]

        return diff_arr
