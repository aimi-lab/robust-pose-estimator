from scipy.spatial.transform import Rotation


def euler2quat(x, y, z, deg_opt: bool = False):

    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler('xyz', [x, y, z], degrees=deg_opt)

    # Convert to quaternions and print
    qs = rot.as_quat()

    return qs


def quat2euler(qs, deg_opt: bool = False):

    rot = Rotation.from_quat(qs)

    rvec = rot.as_euler('xyz', degrees=deg_opt)

    return rvec


def quat2rmat(qs):

    rot = Rotation.from_quat(qs)

    rmat = rot.as_matrix()

    return rmat
