import numpy as np


def forward_project(opts, kmat, rmat=None, tvec=None):

    # init values potentially missing
    rmat = np.eye(3) if rmat is None else rmat
    tvec = np.zeros([3, 1]) if tvec is None else tvec
    opts = np.vstack([opts, np.ones(opts.shape[1])]) if opts.shape[0] == 3 else opts

    # compose projection matrix
    pmat = compose_projection_matrix(kmat, rmat, tvec)

    # pinhole projection
    ipts = pmat @ opts

    # inhomogenization
    ipts = ipts[:3] / ipts[-1]

    return ipts


def reverse_project(ipts, kmat, rmat=None, tvec=None, disp=None, base=float(1)):

    # init values potentially missing
    rmat = np.eye(3) if rmat is None else rmat
    tvec = np.zeros([3, 1]) if tvec is None else tvec
    ipts = np.vstack([ipts, np.ones(ipts.shape[1])]) if ipts.shape[0] == 2 else ipts

    # compose projection matrix
    pmat = compose_projection_matrix(kmat, rmat, tvec)

    # pinhole projection
    opts = np.linalg.pinv(pmat) @ ipts

    # depth assignment
    if disp is not None:
        opts = base * np.diag([kmat[0][0], kmat[1][1], (kmat[0][0]+kmat[1][1])/2]) @ opts[:3] / disp

    return opts


def compose_projection_matrix(kmat, rmat, tvec):
    return kmat @ np.hstack([rmat, tvec])


def decompose_projection_matrix(pmat, scale=False):
    """
    https://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_KR_from_P.m
    """

    n = pmat.shape[0] if len(pmat.shape) == 2 else np.sqrt(pmat.size)
    hmat = pmat.reshape(n, -1)[:, :n]
    rmat, kmat = np.linalg.qr(hmat, mode='reduced')

    if scale:
        kmat = kmat / kmat[n-1, n-1]
        if kmat[0, 0] < 0:
            D = np.diag([-1, -1, *np.ones(n-2)])
            kmat = np.dot(D, kmat)
            rmat = np.dot(rmat, D)

    tvec = np.linalg.lstsq(-pmat[:, :n], pmat[:, -1])[0][:, np.newaxis] if pmat.shape[1] == 4 else np.zeros(n)

    return rmat, kmat, tvec


def create_img_coords(y:int=720, x:int=1280):
    
    x_coords = np.arange(0, x)
    y_coords = np.arange(0, y)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten(), np.ones(len(x_mesh.flatten()))])

    return ipts
