import numpy as np
from scipy.optimize import least_squares

from alley_oop.pose.euler_angles import euler2mat


class FeatPoseEstimator(object):
    """ pose regression based on feature points and non-linear least squares solver """
    def __init__(self, feat_refer=None, feat_query=None, confidence=None, dims_wghts=None, loss_param=None):
        super().__init__()

        # inputs
        self.feat_refer = feat_refer
        self.feat_query = feat_query
        self.confidence = confidence
        self.dimensions = self.feat_refer.shape[0]
        self.feat_wdims = np.ones([self.dimensions])/self.dimensions if dims_wghts is None else dims_wghts
        self.loss_param = float(1) if loss_param is None else loss_param

        # outputs
        self.tvec = np.zeros([3, 1])
        self.rvec = np.zeros([3, 1])
        self.p_list = []
        self.p_star = float('NaN')
        self.p_loss = float('NaN')

    def estimate(self, p_init=None, feat_refer=None, feat_query=None, confidence=None, dims_fit: bool=False):

        # update variables if args available
        self.feat_refer = self.feat_refer if feat_refer is None else feat_refer
        self.feat_query = self.feat_query if feat_query is None else feat_query
        self.confidence = self.confidence if confidence is None else confidence

        # initial guess
        self.tvec = np.mean(self.feat_refer[:3], axis=-1) - np.mean(self.feat_query[:3], axis=-1)
        p_init = self.tvec.flatten().tolist() + self.rvec.flatten().tolist() if p_init is None else p_init
        if dims_fit: p_init += self.feat_wdims.flatten().tolist()

        # compute 6-DOF solution using least squares solver
        self.p_star = least_squares(self.residual_fun, p_init, jac='2-point', args=(dims_fit,), method='lm', max_nfev=int(1e4)).x
        self.p_loss = self.residual_fun(p=self.p_star).sum()

        # assign solution to output vectors
        self.tvec = self.p_star[0:3][np.newaxis].T
        self.rvec = self.p_star[3:6]

        return self.tvec, self.rvec

    def residual_fun(self, p, dims_fit: bool=False):

        # assign current estimate
        self.p_list.append(p)
        tvec = p[0:3][np.newaxis].T
        rmat = euler2mat(*p[3:6])

        # map points from query position
        mpts = rmat @ self.feat_refer[:3] + tvec
        if self.feat_refer.shape[0] > 3: mpts = np.vstack([mpts, self.feat_refer[3:]])

        # squared difference
        sdif = self.compute_diff(mpts, self.feat_query)**2

        # update weights for dimensions (e.g. x, y, z, ...)
        if dims_fit:
            p[-self.dimensions:] = abs(p[-self.dimensions:]) / sum(abs(p[-self.dimensions:]))
            self.feat_wdims = p[-self.dimensions:]

        # reduce space dimensions using weights
        wsqr = self.feat_wdims @ sdif

        # compute loss (huber less sensitive to outliers)
        residuals = self.huber_loss(wsqr, delta=self.loss_param)

        if self.confidence is not None:
            residuals *= self.confidence
        
        return residuals

    def compute_diff(self, rpts, qpts):
        return rpts - qpts

    @staticmethod
    def huber_loss(a, delta: float=1.):

        loss = np.zeros(a.shape)
        idcs = a < delta 
        loss[idcs] = (a[idcs]**2)/2
        loss[~idcs] = delta*(np.abs(a[~idcs]) - delta/2)

        return loss

    @property
    def rmat(self):
        return euler2mat(*self.rvec)
