from macros import *
from manipulator import Joint
import matplotlib.pyplot as plt
import numpy as np
from pytransform3d.rotations import euler_zyx_from_matrix
from scipy.linalg import expm
from utils import parse_data, set_axes_equal


if __name__ == '__main__':
    data = parse_data('data/data_calibrate.txt', animate=False)[0]

    shoulder = None
    elbow = None
    wrist = None
    for point in data:
        idx = int(point[0])
        if idx == SHOULDER_RIGHT:
            shoulder = Joint(point[2], point[3], point[0])
        elif idx == ELBOW_RIGHT:
            elbow = Joint(point[2], point[3], point[0])
        elif idx == WRIST_RIGHT:
            wrist = Joint(point[2], point[3], point[0])

    euler_z, euler_y, euler_x = euler_zyx_from_matrix(
        elbow.t_inv.dot(wrist.t)[:3, :3])
    print('OBSERVED WRIST R:\n', wrist.p)
    print('WRIST R ROTATION:\n', wrist.t)
    print('RELATIVE TRANSFORM:\n', elbow.t_inv.dot(wrist.t))
    translation = elbow.t_inv.dot(
        np.array([wrist.p[0], wrist.p[1], wrist.p[2], 1]))
    print(translation)
    transform = \
        expm(np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 0, 0],
            [0,  0, 0, 0],
        ]) * -euler_z).dot(
        expm(np.array([
            [0,  0, 1, 0],
            [0,  0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 0],
        ]) * -euler_y)).dot(
        expm(np.array([
            [0, 0,  0, 0],
            [0, 0, -1, 0],
            [0, 1,  0, 0],
            [0, 0, 0, 0],
        ]) * -euler_x))
    transform = np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1],
    ]).dot(transform)
    print('EXPECTED RELATIVE TRANSFORM:\n', transform)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(elbow.p[0], elbow.p[1], elbow.p[2], color='r')
    ax.scatter(wrist.p[0], wrist.p[1], wrist.p[2], color='b')
    set_axes_equal(ax)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.invert_zaxis()
    ax.view_init(azim=270, elev=105, )
    plt.show()
