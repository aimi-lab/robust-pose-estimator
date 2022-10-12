import torch
import gpytorch
from alley_oop.geometry.lie_3d import lie_se3_to_SE3_batch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class ExactGP(gpytorch.models.ExactGP):
    """
        Batch-independent Multi-task Gaussian-Process Regressor

        We use batch-independent multi-output GPs with RBF-Kernels. We have O independent GPs (in the batch dimension)
        with the same C-dimensional inputs.
        We use input-dependent Gaussian noise as a likelihood model.
        Note that it is possible (but not implemented) to have multi-output GPs with task dependant kernels.
    """

    def __init__(self,  length_scale: float=0.1):
        """

        :param length_scale: RBF kernel length-scale parameter
        :param noise_level: Noise variance
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood()  # dummy likelihood
        super(ExactGP, self).__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([3]))  # no deformation far from observed points
        rbf_kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3]))
        rbf_kernel.initialize(lengthscale=length_scale)
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf_kernel, batch_shape=torch.Size([3]))
        self.eval()

    def fit(self, train_x: torch.tensor, train_y: torch.tensor, noise_level: torch.tensor):
        """
        :param train_x: training samples with shape N x C (N: number of samples, C: number of input dimensions)
        :param train_y: target observations with shape O x N (N: number of samples, O: number of output dimensions)
        :param noise_level: target observations noise with shape O x N (N: number of samples, O: number of output dimensions)
        """
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise_level)
        self.set_train_data(train_x, train_y, strict=False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_WarpFieldEstimator(torch.nn.Module):
    """
        Warp-field estimation with GP-Regression in se(3)
    """
    def __init__(self, length_scale: float=0.1, noise_level: float=0.001):
        """
        :param length_scale: RBF kernel length-scale parameter
        :param noise_level: min noise variance
        """
        super(GP_WarpFieldEstimator, self).__init__()
        self.gp_regressor = ExactGP(length_scale)
        self.noise_level = noise_level

    def fit(self, ref_opts: torch.tensor, target_opts: torch.tensor, noise: torch.tensor=None):
        """
        :param ref_opts: reference 3d points, shape 3xN
        :param target_opts: target 3d points, shape 3xN

        """
        # ref_pcl and target_pcl need to be registered already: ref_plc.opts[0] <-> target_pcl.opts[0]
        # lie-algebra translation:
        def_field_se3 = target_opts - ref_opts

        # map the depth confidence to a noise prior
        noise = torch.ones_like(def_field_se3, dtype=torch.float16) if noise is None else noise
        self.gp_regressor.fit(ref_opts.T, def_field_se3, noise_level=noise*self.noise_level)

    def predict(self, ref_opts: torch.tensor):
        """
            :param ref_opts: reference 3d points, shape 3xn
            :return: Warp-field as nx4x4 homogeneous transforms
        """
        deformation_field_se3 = self.gp_regressor(ref_opts.T).loc
        return deformation_field_se3


class GP_WarpFieldEstimator_sklearn(torch.nn.Module):
    """
        Gaussian-Process Warp Field estimation
    """
    def __init__(self, length_scale: float=0.1, noise_level: float=0.001):
        super(GP_WarpFieldEstimator_sklearn, self).__init__()
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        self.gp_regressor = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    def fit(self, ref_opts: torch.tensor, ref_nrml: torch.tensor, target_opts: torch.tensor, target_nrml:torch.tensor):
        """
        :param ref_pcl: shape at rest surfel map
        :param target_pcl: deformed surfel map

        """
        # ref_pcl and target_pcl need to be registered already: ref_plc.opts[0] <-> target_pcl.opts[0]
        # lie-algebra rotation:
        wvecs = torch.linalg.cross(target_nrml, ref_nrml, dim=0)
        # lie-algebra translation:
        tvecs = target_opts - ref_opts
        def_field_se3 = torch.cat([wvecs, tvecs]).T.cpu().numpy()
        self.gp_regressor.fit(ref_opts.T.cpu().numpy(), def_field_se3)

    def predict(self, ref_opts: torch.tensor):
        """
            :param def_pcl: optional surfel map to predict warp-field (if not provided use ref_pcl)
            :return: Warp-field as Nx4x4 homogenious transforms
        """
        deformation_field_se3 = self.gp_regressor.predict(ref_opts.T.cpu().numpy())
        deformation_field_SE3 = lie_se3_to_SE3_batch(torch.tensor(deformation_field_se3, device=ref_opts.device)).float()
        return deformation_field_SE3