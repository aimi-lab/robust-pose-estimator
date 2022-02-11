import numpy as np


def align(reference, query, estimate_scale=False):
    """Align two trajectories using the method of Horn (closed-form).
    B. K. P. Horn, “Closed-form solution of absolute orientation using unit quaternions,”
    Journal of the Optical Society of America A , vol. 4, no. 4, pp. 629–642, 1987

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    model_zerocentered = reference - reference.mean(1)[:,None]
    data_zerocentered = query - query.mean(1)[:,None]

    W = np.zeros((3, 3))
    for column in range(reference.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    scale = float(dots / norms) if estimate_scale else 1.0

    trans = query.mean(1)[:,None] - scale * rot * reference.mean(1)[:,None]

    model_aligned = scale * rot * reference + trans
    alignment_error = model_aligned - query

    residuals = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, residuals, scale
