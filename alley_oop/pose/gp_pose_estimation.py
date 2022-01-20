import numpy as np
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from alley_oop.pose.euler_angles import mat2euler
from alley_oop.pose.feat_pose_estimation import FeatPoseEstimator


class GpPoseEstimator(FeatPoseEstimator):
    """ pose regression based on feature points and non-linear least squares solver """

    def __init__(self, feat_refer=None, feat_query=None, confidence=None, dims_wghts=None, loss_param=None, length_scale=1.0, noise_level=1.0, max_iter=10):
        super().__init__(feat_refer, feat_query, confidence, dims_wghts, loss_param)
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        self.gp_regressor = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        self.max_iter = max_iter

    def estimate(self, p_init=None, feat_refer=None, feat_query=None, confidence=None, dims_fit: bool = False):
        self.feat_refer = self.feat_refer if feat_refer is None else feat_refer
        self.feat_query = self.feat_query if feat_query is None else feat_query
        self.confidence = self.confidence if confidence is None else confidence

        feat_refer_deformed = self.feat_refer
        initial_feat_refer = self.feat_refer.copy()

        for i in range(self.max_iter): #until convergence
            # solve first rigid transform (we want to find the inverse because then we don't have to recompute the GP at each iteration)
            super().estimate(p_init, feat_refer_deformed, self.feat_query, confidence, dims_fit)
            # rigidly transform query points
            feat_query_transformed = self.inv_rigid_transform(self.feat_query)
            # estimate deformation using GP
            self.gp_regressor.fit(initial_feat_refer, feat_query_transformed)
            feat_refer_deformed = self.gp_regressor.predict(initial_feat_refer)
            # use last estimate of parameter as initialization for new estimation
            p_init = np.concatenate((self.tvec.squeeze(1), mat2euler(self.rmat)))
            # stop criterion
            if self.p_loss < 1e-5:
                break

        return self.tvec, self.rvec

    def rigid_transform(self, feat_refer=None):
        if feat_refer is None:
            feat_refer = self.feat_refer

        # map points from query position
        mpts = self.rmat @ feat_refer[:3] + self.tvec
        if feat_refer.shape[0] > 3: mpts = np.vstack([mpts, feat_refer[3:]])
        return mpts

    def inv_rigid_transform(self, feat_query=None):
        if feat_query is None:
            feat_query = self.feat_query

        # map points from query position
        mpts = self.rmat.T @ (feat_query[:3] - self.tvec)
        if feat_query.shape[0] > 3: mpts = np.vstack([mpts, feat_query[3:]])
        return mpts

    def deform(self, feat_refer=None):
        if feat_refer is None:
            feat_refer = self.feat_query
        # apply deformation
        feat_refer_deformed = self.gp_regressor.predict(feat_refer)
        # apply rigid transform
        feat_refer_deformed = self.rigid_transform(feat_refer_deformed)
        return feat_refer_deformed


reference = np.array([[0,0,0], [100,0,0], [0,0,1], [1,1,1], [1,1,0], [1,0,1], [0,1,1], [10, 4,1], [100, 100, 100]]).T
from scipy.spatial.transform import Rotation as R
R_true = R.from_euler('x', 90, degrees=True).as_matrix()
t_true = np.array([1,1,1])[:,None]

query = R_true @ reference + t_true
noise_level = 0.001
query_noisy = query + noise_level*np.random.randn(*query.shape)

estimator = FeatPoseEstimator(feat_query=query_noisy, feat_refer=reference)
estimator.estimate()
reference_deformed = estimator.rmat @ reference[:3] + estimator.tvec
print('residuals: ', np.sum((reference_deformed- query)**2))

estimator = GpPoseEstimator(feat_query=query_noisy, feat_refer=reference, length_scale=10, noise_level=noise_level**2)
estimator.estimate()
print(estimator.rmat-R_true, estimator.tvec-t_true)
reference_deformed = estimator.deform(reference)
print('residuals: ', np.sum((reference_deformed- query)**2))
from alley_oop.utils.absolute_pose_quarternion import align

#print(align(reference, query))