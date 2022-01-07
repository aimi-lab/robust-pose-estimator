import numpy as np

def euler2mat(theta_x: float = 0, theta_y: float = 0, theta_z: float = 0):
    """
    Creation of a rotation matrix from three angles in radians
    """

    # matrix for counter-clockwise rotation around x-y-z axes
    rmat_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]]
                        )

    rmat_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]]
                        )

    rmat_x = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]]
                        )

    rmat_3 = np.dot(np.dot(rmat_x, rmat_y), rmat_z)

    return rmat_3

def mat2euler(rmat):
    """
    https://www.geometrictools.com/Documentation/EulerAngles.pdf
    """

    if rmat[0, 2] < 1:
        if rmat[0, 2] > -1:
            theta_y = np.arcsin(rmat[0, 2])
            theta_x = np.arctan2(-rmat[1, 2], rmat[2, 2])
            theta_z = np.arctan2(-rmat[0, 1], rmat[0, 0])
        else:
            theta_y = -np.pi / 2
            theta_x = -np.arctan2(rmat[1, 0], rmat[1, 1])
            theta_z = 0
    else:
        theta_y = np.pi / 2
        theta_x = np.arctan2(rmat[1, 0], rmat[1, 1])
        theta_z = 0

    return np.array([theta_x, theta_y, theta_z])
