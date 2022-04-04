import torch
from torch.nn.functional import conv2d
import numpy as np
from torchgeometry import homography_warp

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


def proj_jacobian(x,y):
    """ we combine Jpi, Jw and JH(0) in a single matrix 2x8, this can be found by multiplying the individual matrices in
        https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf.
        Note that we set the focal length f=1 since it is a common factor and let Zs=1 for the image plane at 1,
        we adapt J(H0) for the special case of rotation with three skew-symmetric matrices as generators for so(3)

        this leads to [xy         -1-x**2       y
                       -1+y**2     -xy          x]"""

    return torch.tensor([[x*y,-1-x**2, y],
                        [-1+y**2, -x*y, x]])


def batch_proj_jacobian(img):
    """ we combine Jpi, Jw and JH(0) in a single matrix 2x8, this can be found by multiplying the individual matrices in
        https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf.
        Note that we set the focal length f=1 since it is a common factor and let Zs=1 for the image plane at 1,
        we adapt J(H0) for the special case of rotation with three skew-symmetric matrices as generators for so(3)

        this leads to [xy         -1-x**2       y
                       -1+y**2     -xy          x]"""
    xs, ys = torch.meshgrid(torch.arange(img.shape[-1]), torch.arange(img.shape[-2]))
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    # J = torch.empty((img.shape[-1]*img.shape[-2], 2, 3))
    # # slow loop
    # for i, (x,y) in enumerate(zip(xs, ys)):
    #     J[i] = proj_jacobian(x,y)

    # fast
    J2 = torch.empty((img.shape[-1] * img.shape[-2], 2, 3))
    J2[:, 0, 0] = xs*ys
    J2[:, 0, 1] = -xs*xs-1
    J2[:, 0, 2] = ys
    J2[:, 1, 0] = ys*ys-1
    J2[:, 1, 1] = -xs*ys
    J2[:, 1, 2] = xs

    return J2


def ems_jacobian(img1, img2, batch_proj_jac=None):
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
        batch_proj_jac = batch_proj_jacobian(img1)
    J = J_img @ batch_proj_jac
    return J.squeeze(1)


def so3(x):
    phi = torch.sqrt(torch.dot(x,x))
    w_x = x[0]*G1
    return torch.eye(3) + torch.sin(phi)/phi

def warp_img(img, R, K):
    import cv2
    img2cv = cv2.warpPerspective(img.numpy(), (K @ R@ torch.linalg.inv(K)).numpy().squeeze(), (img.shape[1], img.shape[0]))
    return torch.tensor(img2cv)


def efficient_least_squares(img1, img2, K, n_iter=100, res_thr=0.0001):
    proj_jac = batch_proj_jacobian(img1)

    R_lr = torch.eye(3).unsqueeze(0)
    for i in range(n_iter):
        # compute residuals f(x)
        warped_img = warp_img(img1, R_lr, K)
        J = ems_jacobian(warped_img, img2, proj_jac)
        J_pinv = torch.linalg.pinv(J)
        residuals = (0.5*(warped_img - img2)**2).view(-1, 1)
        # compute update parameter x0
        x0 = -J_pinv @ residuals

        # update rotation estimate
        R_lr = R_lr @ so3(x0)
        if residuals.mean() < res_thr
            break
    return R_lr, residuals, warped_img

from scipy.spatial.transform import Rotation
img1 = torch.rand((100,120)).float()
f=1200.0
cx=60
cy=50
intrinsics = torch.tensor([[f, 0, cx], [0, f, cy], [0,0,1.0]]).float()
R_true = torch.tensor(Rotation.from_euler('z', 10.0, degrees=True).as_matrix()).float()
img2cv = warp_img(img1, R_true, intrinsics)
def plot(img1, img2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img1.squeeze())
    ax[1].imshow(img2.squeeze())
    plt.show()
plot(img1, img2cv)
R, residuals, warped_img = efficient_least_squares(img1, img2cv, intrinsics)
plot(warped_img, img2cv)






