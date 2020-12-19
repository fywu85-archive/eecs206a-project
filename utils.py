import matplotlib.pyplot as plt
from manipulator import Joint
from macros import *
import numpy as np

plt.rcParams['font.family'] = 'FreeSans'
plt.rcParams['font.size'] = 12


def parse_data(filename, animate=False):
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
            # if idx in JOINTS_OF_INTEREST:
            t = float(line[3])
            p = (float(line[11]), -float(line[12]), float(line[13]))
            q = (float(line[17]), float(line[18]),
                 float(line[19]), float(line[20]))
            data.append([idx, t, p, q])

    data = np.array(data, dtype=object)
    data[:, 1] = data[:, 1] - data[0, 1]
    data = np.reshape(data, (-1, 32, 4))

    if animate:
        animate_data(data)

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

    plot_radius = 0.5*max([x_range, y_range, z_range])
    handle.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    handle.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    handle.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def animate_data(data):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for idx, frame in enumerate(data):
        ax.clear()
        skeleton = {}
        for point in frame:
            joint = Joint(point[2], point[3], point[0])
            skeleton[joint.index] = joint
            if joint.index in JOINTS_OF_INTEREST:
                color = 'm'
                # ax.text(joint.p[0], joint.p[1], joint.p[2],
                #         JOINT_INDICES[joint.index])
                ax.text(
                    joint.p[0], joint.p[1], joint.p[2],
                    '({:.1f}, {:.1f}, {:.1f})'.format(
                        joint.e[0], joint.e[1], joint.e[2]))
                ax.plot(
                    [joint.p[0], joint.p[0] + joint.r[0, 0] * 100],
                    [joint.p[1], joint.p[1] + joint.r[1, 0] * 100],
                    [joint.p[2], joint.p[2] + joint.r[2, 0] * 100], color='r')
                ax.plot(
                    [joint.p[0], joint.p[0] + joint.r[0, 1] * 100],
                    [joint.p[1], joint.p[1] + joint.r[1, 1] * 100],
                    [joint.p[2], joint.p[2] + joint.r[2, 1] * 100], color='y')
                ax.plot(
                    [joint.p[0], joint.p[0] + joint.r[0, 2] * 100],
                    [joint.p[1], joint.p[1] + joint.r[1, 2] * 100],
                    [joint.p[2], joint.p[2] + joint.r[2, 2] * 100], color='b')
            else:
                color = 'c'
            ax.scatter(joint.p[0], joint.p[1], joint.p[2], color=color)
        for child, parent in SKELETON_INDICES.items():
            p1 = skeleton[child].p
            p2 = skeleton[parent].p
            if child in JOINTS_OF_INTEREST and \
                    parent in JOINTS_OF_INTEREST:
                color = 'm'
            else:
                color = 'c'
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]], color=color)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_xlim3d([-800, 800])
        ax.set_ylim3d([-1000, 600])
        ax.set_zlim3d([500, 2100])
        set_axes_equal(ax)
        ax.invert_zaxis()
        ax.view_init(azim=270, elev=105, )
        plt.tight_layout()
        # plt.pause(0.05)
        plt.show()
        # plt.savefig('animate/%05d.png' % idx, bbox_inches=None)


def negate_hist(hist, bins):
    hist = -1 * hist
    hist = hist - min(hist)
    hist = hist / np.sum(hist * np.diff(bins))
    return hist, bins


def sample_hist(hist, bins, num=1):
    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    #np.random.seed(205)
    value = np.random.rand(num)
    value_bin = np.searchsorted(cdf, value)
    sample = bin_midpoints[value_bin] #+ \
    #    np.diff(bins)[value_bin] * (np.random.rand() - 0.5)
    return sample


def render(manipulator, thetas, data):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    skeleton = {}
    ARM_RIGHT = JOINTS_OF_INTEREST + [HAND_RIGHT, HANDTIP_RIGHT, THUMB_RIGHT]
    for point in data[0]:
        joint = Joint(point[2], point[3], point[0])
        if joint.index not in ARM_RIGHT:
            skeleton[joint.index] = joint.p
            ax.scatter(joint.p[0], joint.p[1], joint.p[2], color='c')

    for idx, g in enumerate(manipulator.forward_kinematic(thetas)):
        point = g[:-1, -1]
        skeleton[ARM_RIGHT[idx]] = point
        if idx < 5:
            ax.scatter(point[0], point[1], point[2], color='m')
        else:
            ax.scatter(point[0], point[1], point[2], color='c')

    for child, parent in SKELETON_INDICES.items():
        p1 = skeleton[child]
        p2 = skeleton[parent]
        if child in JOINTS_OF_INTEREST and \
                parent in JOINTS_OF_INTEREST:
            color = 'm'
        else:
            color = 'c'
        ax.plot([p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]], color=color)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_xlim3d([-800, 800])
    ax.set_ylim3d([-1000, 600])
    ax.set_zlim3d([500, 2100])
    set_axes_equal(ax)
    ax.invert_zaxis()
    ax.view_init(azim=270, elev=105, )
    # plt.show()

