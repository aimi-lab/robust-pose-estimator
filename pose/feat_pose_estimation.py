import numpy as np
from scipy.optimize import least_squares

from pose.euler_angles import euler2mat


class FeatPoseEstimator(object):
    """ pose regression based on feature points and non-linear least squares solver """
    def __init__(self, feat_refer=None, feat_query=None, confidence=None):
        super().__init__()

        # inputs
        self.feat_refer = feat_refer
        self.feat_query = feat_query
        self.confidence = confidence

        # outputs
        self.tvec = np.zeros([3, 1])
        self.rvec = np.zeros([3, 1])
        self.p_list = []

    def estimate(self, p_init=None, feat_refer=None, feat_query=None, confidence=None, dim_weights=None):

        # update variables if args available
        self.feat_refer = self.feat_refer if feat_refer is None else feat_refer
        self.feat_query = self.feat_query if feat_query is None else feat_query
        self.confidence = self.confidence if confidence is None else confidence
        p_init = self.tvec.flatten().tolist() + self.rvec.flatten().tolist() if p_init is None else p_init

        # compute 6-DOF solution using least squares solver
        p_star = least_squares(self.residual_fun, p_init, jac='2-point', args=(dim_weights,), method='lm').x

        # assign solution to output vectors
        self.tvec = p_star[0:3][np.newaxis].T
        self.rvec = p_star[3:6]

        return self.tvec, self.rvec

    def residual_fun(self, p, dim_weights=None):

        dims = self.feat_refer.shape[0]
        dim_weights = np.ones(dims)/dims if dim_weights is None else dim_weights

        # assign current estimate
        self.p_list.append(p)
        tvec = p[0:3][np.newaxis].T
        rmat = euler2mat(*p[3:6])

        # map points from query position
        mpts = rmat @ self.feat_query + tvec

        # squared difference
        sdif = (self.feat_refer-mpts)**2

        # reduce space dimensions using weights
        wsqr = dim_weights @ sdif

        # compute loss (huber less sensitive to outliers)
        residuals = self.huber_loss(wsqr)

        print(residuals.sum())

        if self.confidence is not None:
            residuals *= self.confidence
        
        return residuals

    @staticmethod
    def huber_loss(a, delta=1):

        loss = np.zeros(a.shape)
        idcs = a < delta 
        loss[idcs] = (a[idcs]**2)/2
        loss[~idcs] = delta*(np.abs(a[~idcs]) - delta/2)

        return loss

    @property
    def rmat(self):
        return euler2mat(*self.rvec)
