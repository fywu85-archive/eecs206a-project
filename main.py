from macros import *
from manipulator import Joint, Manipulator
import numpy as np
from utils import parse_data, negate_hist, sample_hist
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from utils import render


if __name__ == '__main__':
    data = parse_data('data/data_random.txt', animate=False)
    buffer = {}
    for index in JOINTS_OF_INTEREST:
        buffer[index] = []
    for frame in data:
        for point in frame:
            joint = Joint(point[2], point[3], point[0])
            if joint.index in JOINTS_OF_INTEREST:
                buffer[joint.index].append([point[1]/1000.] + list(joint.e))

    arm_R = Manipulator('ARM R', data)
    render(arm_R, np.zeros(12), data)

    thetas = []
    for index in JOINTS_OF_INTEREST[:-1]:
        print(JOINT_INDICES[index])
        buffer_data = np.array(buffer[index])
        t = buffer_data[:, 0]
        euler_z = UnivariateSpline(t, buffer_data[:, 1])
        euler_y = UnivariateSpline(t, buffer_data[:, 2])
        euler_x = UnivariateSpline(t, buffer_data[:, 3])
        fig = plt.figure()
        for idx in range(3):
            ax = fig.add_subplot(3, 1, idx+1)
            ax.step(t, euler_z.derivative(n=idx)(t), label='Euler Z')
            ax.step(t, euler_y.derivative(n=idx)(t), label='Euler Y')
            ax.step(t, euler_x.derivative(n=idx)(t), label='Euler X')
            ax.legend()
            if idx == 0:
                ax.set_title(JOINT_INDICES[index])
                ax.set_ylabel('Angle [rad]')
            elif idx == 1:
                ax.set_ylabel('Ang. Vel. [rad/s]')
            else:
                ax.set_ylabel('Ang. Acc. [rad/s2]')
                ax.set_xlabel('Time [s]')
        fig.tight_layout()
        plt.savefig(JOINT_INDICES[index] + 'timeseries.png',
                    bbox_inches=None, dpi=300)

        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.hist(euler_z(t), bins=30, density=True)
        hist, bins = np.histogram(euler_z(t), bins=30, density=True)
        hist, bins = negate_hist(hist, bins)
        ax.step(bins, np.array([hist[0]] + list(hist)), label='recommend')
        # sample = sample_hist(hist, bins, 100000)
        # ax.hist(sample, bins=30, density=True)
        thetas.append(sample_hist(hist, bins, 10))
        ax.set_title(JOINT_INDICES[index] + ' Euler Z')
        ax.set_ylabel('Prob. Density')
        ax.legend()

        ax = fig.add_subplot(3, 1, 2)
        ax.hist(euler_y(t), bins=30, density=True)
        hist, bins = np.histogram(euler_y(t), bins=30, density=True)
        hist, bins = negate_hist(hist, bins)
        ax.step(bins, np.array([hist[0]] + list(hist)), label='recommend')
        thetas.append(sample_hist(hist, bins, 10))
        ax.set_title(JOINT_INDICES[index] + ' Euler Y')
        ax.set_ylabel('Prob. Density')
        ax.legend()

        ax = fig.add_subplot(3, 1, 3)
        ax.hist(euler_x(t), bins=30, density=True)
        hist, bins = np.histogram(euler_x(t), bins=30, density=True)
        hist, bins = negate_hist(hist, bins)
        ax.step(bins, np.array([hist[0]] + list(hist)), label='recommend')
        thetas.append(sample_hist(hist, bins, 10))
        ax.set_title(JOINT_INDICES[index] + ' Euler X')
        ax.set_ylabel('Prob. Density')
        ax.set_xlabel('Angle [rad]')
        ax.legend()

        fig.tight_layout()
        plt.savefig(JOINT_INDICES[index] + 'hist.png',
                    bbox_inches=None, dpi=300)

    thetas = np.array(thetas)
    print(thetas.shape)
    thetas[:3, :] = 0
    for idx in range(thetas.shape[1]):
        render(arm_R, thetas[:, idx], data)
        plt.tight_layout()
        plt.savefig('recommend%02d.png' % idx, bbox_inches=None, dpi=300)
