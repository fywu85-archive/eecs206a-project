from macros import *
from manipulator import Manipulator
import matplotlib.pyplot as plt
import numpy as np
from utils import parse_data, set_axes_equal


if __name__ == '__main__':
    data = parse_data('data/data_train_v1.txt')
    arm = Manipulator('ARM R', data)

    interested_joints = [
        SPINE_CHEST,
        CLAVICLE_RIGHT,
        SHOULDER_RIGHT,
        ELBOW_RIGHT
    ]
    theta_frames = []
    wrist_frames = []
    for tf, picture in zip(arm.tf_data, arm.picture_data):
        theta = [
            item
            for index in interested_joints
            for item in tf[index][1].to_list()]
        theta_frames.append(theta)
        wrist_frames.append(picture[WRIST_RIGHT].p)
    theta_frames = np.array(theta_frames)
    wrist_frames = np.array(wrist_frames)
    np.save('data/y_train.npy', theta_frames)
    np.save('data/x_train.npy', wrist_frames[:, :3])

    plot = False
    if plot:
        fig = plt.figure()
        if plot == '2d':
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(wrist_frames[:, 0], wrist_frames[:, 2])
            ax.set_aspect('equal', adjustable='box')
        elif plot == '3d':
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            for x, y, z in zip(wrist_frames[:, 0],
                               wrist_frames[:, 1],
                               wrist_frames[:, 2]):
                ax.scatter(x, y, z)
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Y [mm]')
            ax.set_zlabel('Z [mm]')
            set_axes_equal(ax)
            ax.invert_zaxis()
            ax.view_init(azim=270, elev=105, )
        plt.tight_layout()
        plt.show()
