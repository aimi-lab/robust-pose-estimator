import torch
from torch.nn.functional import conv2d
import numpy as np
from torchgeometry import homography_warp
from alley_oop.geometry.lie_3d import lie_alebra2group_rot, lie_hatmap

""" this is an implementation of  
    https://www.robots.ox.ac.uk/~cmei/articles/omni_track_mei.pdf
    https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf
    
    It estimates the camera rotation between two images (assuming rotation only) using efficient second-order minimization
"""


def image_jacobian(img):
    sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    if img.ndim < 4:
        img = img.unsqueeze(1)
    batch, channels, h, w= img.shape
    sobel_kernelx = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
    sobel_kernely = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3).transpose(2,3)
    x_grad = conv2d(img, sobel_kernelx, stride=1, padding=1, groups=channels).reshape(batch, channels, -1)
    y_grad = conv2d(img, sobel_kernely, stride=1, padding=1, groups=channels).reshape(batch, channels, -1)
    jacobian = torch.stack((x_grad, y_grad), dim=-1)
    return jacobian



def batch_jw(img, K):
    """ jacobian of action w with respect to R(0)
        w<KR(x)K'><(u v 1)T>
        adapted from https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf where H(x) = K R(x) K'
        """
    assert K.shape == (3,3)
    u, v = torch.meshgrid(torch.arange(img.shape[-1]), torch.arange(img.shape[-2]))
    u = u.T.reshape(-1)
    v = v.T.reshape(-1)

    # intrinsics
    cu = K[0,2]
    cv = K[1,2]
    f = K[0,0]

    # fast
    J2 = torch.zeros((img.shape[-1] * img.shape[-2], 3, 9))
    J2[:, 0, 0] = u -cu
    J2[:, 0, 1] = v -cv
    J2[:, 0, 2] = f

    J2[:, 0, 6] = -1/f*(u-cu)**2
    J2[:, 0, 7] = -1/f*(u-cu)*(v-cv)
    J2[:, 0, 8] = -(u -cu)

    J2[:, 1, 3] = u -cu
    J2[:, 1, 4] = v -cv
    J2[:, 1, 5] = f

    J2[:, 1, 6] = -1/f*(u-cu)*(v-cv)
    J2[:, 1, 7] = -1/f*(v-cv)**2
    J2[:, 1, 8] = -(v-cv)

    return J2


def j_rot():
    """ jacobian of R(x) wrt. x -> generators of so(3) in vector form
        adapted from https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf
        """
    J = torch.empty((9,3))
    J[:, 0] = torch.tensor(lie_hatmap(np.array([1,0,0]))).reshape(-1)
    J[:, 1] = torch.tensor(lie_hatmap(np.array([0,1,0]))).reshape(-1)
    J[:, 2] = torch.tensor(lie_hatmap(np.array([0,0,1]))).reshape(-1)
    return J


def ems_jacobian(img1, img2, K, batch_proj_jac=None):
    """ Jacobian for efficient least squares (Jimg1 + Jimg2)/2 * Jproj in R^(h*w)x2"""
    assert img1.ndim == 2
    assert img2.ndim == 2
    h,w = img1.shape
    assert h == img2.shape[0]
    assert w == img2.shape[1]

    J_img = image_jacobian(torch.stack((img1, img2)))

    J_img = (J_img[0] + J_img[1])/2
    J_img = J_img.reshape(h*w,1,2).squeeze(0)
    if batch_proj_jac is None:
        batch_proj_jac = (batch_jw(img1, K) @ j_rot())[:, :2] # remove third line (zeros because we don't use w coordinates)
    J = J_img @ batch_proj_jac
    return J.squeeze(1), batch_proj_jac


def so3(x):
    return torch.tensor(lie_alebra2group_rot(x.numpy().squeeze())).float()

def warp_img(img, R, K):
    import cv2
    img2cv = cv2.warpPerspective(img.numpy(), (K @ R@ torch.linalg.inv(K)).numpy().squeeze(), (img.shape[1], img.shape[0]))
    return torch.tensor(img2cv)


def efficient_least_squares(img1, img2, K, n_iter=1000, res_thr=0.00001):

    R_lr = torch.eye(3).unsqueeze(0)
    J_proj = None
    for i in range(n_iter):
        # compute residuals f(x)
        warped_img = warp_img(img1, R_lr, K)
        J, J_proj = ems_jacobian(warped_img, img2, K, J_proj)
        J_pinv = torch.linalg.pinv(J)
        residuals = (0.5*(warped_img - img2)**2).reshape(-1, 1)
        # compute update parameter x0
        x0 = -J_pinv @ residuals

        # update rotation estimate
        R_lr = R_lr @ so3(x0)
        print(x0)
        print(residuals.mean())
        plot(warped_img, img2)

        if residuals.mean() < res_thr:
            break
    return R_lr, residuals, warped_img

from scipy.spatial.transform import Rotation
import cv2
img1 = torch.tensor(cv2.resize(cv2.imread('../../tests/test_data/000000l.png', cv2.IMREAD_GRAYSCALE), (160, 128))).float()
f=1200.0
cx=79.5
cy=63.5
intrinsics = torch.tensor([[f, 0, cx], [0, f, cy], [0,0,1.0]]).float()
R_true = torch.tensor(Rotation.from_euler('z', 3.0, degrees=True).as_matrix()).float()
img2cv = warp_img(img1, R_true, intrinsics)
def plot(img1, img2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img1.squeeze())
    ax[1].imshow(img2.squeeze())
    plt.show()
plot(img1, img2cv)
R, residuals, warped_img = efficient_least_squares(img1, img2cv, intrinsics)
print(R)
print(residuals)
plot(warped_img, img2cv)






