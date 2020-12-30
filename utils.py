import matplotlib.pyplot as plt
import numpy as np
from pytransform3d.rotations import matrix_from_quaternion
from pytransform3d.rotations import euler_zyx_from_matrix
from pytransform3d.rotations import quaternion_from_matrix
from scipy.linalg import expm

plt.rcParams['font.family'] = 'FreeSans'
plt.rcParams['font.size'] = 12
np.random.seed(205)


class Joint:
    def __init__(
            self, p=np.zeros((3,)), q=np.zeros((4,)), index='', reflect=True):
        self.index = index
        self.p = np.ones((4,))
        self.p[:3] = np.array(p) / 1000
        self.q = np.array(q)
        rotation_matrix = matrix_from_quaternion(self.q)
        if reflect:
            y180 = expm(np.array([
                [0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0],
            ]) * np.pi)
            self.r = y180.dot(rotation_matrix)
        else:
            self.r = rotation_matrix
        self.e = euler_zyx_from_matrix(self.r)
        self.t = np.zeros((4, 4))
        self.t[:3, :3] = self.r
        self.t[:, 3] = self.p
        self.t_inv = np.linalg.inv(self.t)
        self.reflect = reflect

    def update(self, p, q):
        self.p = np.ones((4,))
        self.p[:3] = np.array(p) / 1000
        self.q = np.array(q)
        rotation_matrix = matrix_from_quaternion(self.q.as_quat())
        if self.reflect:
            y180 = expm(np.array([
                [0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0],
            ]) * np.pi)
            self.r = y180.dot(rotation_matrix)
        else:
            self.r = rotation_matrix
        self.e = euler_zyx_from_matrix(self.r)
        self.t = np.zeros((4, 4))
        self.t[:3, :3] = self.r
        self.t[:, 3] = self.p
        self.t_inv = np.linalg.inv(self.t)


class Euler:
    def __init__(self, z, y, x):
        self.z = z
        self.y = y
        self.x = x


def compute_rotation(parent, child):
    z, y, x = euler_zyx_from_matrix(
        parent.t_inv.dot(child.t)[:3, :3])
    return Euler(z, y, x)


def compute_translation(parent, child):
    trans = parent.t_inv.dot(
        np.array([child.p[0], child.p[1], child.p[2], 1]))
    return trans


def compute_transform(parent, trans, euler):
    if isinstance(parent, np.ndarray):
        parent = transform_to_joint(parent)
    transform = \
        expm(np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]) * -euler.z).dot(
            expm(np.array([
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 0],
            ]) * -euler.y)).dot(
            expm(np.array([
                [0, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]) * -euler.x))
    transform = parent.t.dot(np.array([
        [1, 0, 0, trans[0]],
        [0, 1, 0, trans[1]],
        [0, 0, 1, trans[2]],
        [0, 0, 0, 1],
    ]).dot(transform))
    return transform


def transform_to_joint(tf):
    p = tf[:3, 3] * 1000
    rotation_matrix = tf[:3, :3]
    y180 = expm(np.array([
        [0, 0, 1],
        [0, 0, 0],
        [-1, 0, 0],
    ]) * np.pi)
    rotation_matrix = y180.dot(rotation_matrix)
    q = quaternion_from_matrix(rotation_matrix)
    return Joint(p=p, q=q, reflect=True)


def negate_histogram(hist, bins):
    hist = -1 * hist
    hist = hist - min(hist)
    hist = hist / np.sum(hist * np.diff(bins))
    return hist, bins


def sample_histogram(hist, bins, num=1):
    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    value = np.random.rand(num)
    value_bin = np.searchsorted(cdf, value)
    sample = bin_midpoints[value_bin]  # + \
    #    np.diff(bins)[value_bin] * (np.random.rand() - 0.5)
    return sample


def parse_data(filename):
    data = []
    with open(filename, 'r') as fp:
        for line in fp:
            line = line.replace(';', '')
            line = line.replace(',', '')
            line = line.replace('[', '[ ')
            line = line.replace(']', ' ]')
            line = line.replace('(', '( ')
            line = line.replace(')', ' )')
            line = line.split()
            idx = int(line[5])
            t = float(line[3])
            p = (float(line[11]), -float(line[12]), float(line[13]))
            q = (float(line[17]), float(line[18]),
                 float(line[19]), float(line[20]))
            data.append([idx, t, p, q])
    data = np.array(data, dtype=object)
    data[:, 1] = data[:, 1] - data[0, 1]
    data = np.reshape(data, (-1, 32, 4))
    return data


def set_axes_equal(handle):
    x_limits = handle.get_xlim3d()
    y_limits = handle.get_ylim3d()
    z_limits = handle.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    handle.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    handle.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    handle.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
