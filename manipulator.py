from macros import *
import matplotlib.pyplot as plt
import numpy as np
from utils import Joint, Euler
from utils import compute_rotation, compute_translation, compute_transform
from utils import set_axes_equal
from utils import transform_to_joint
from utils import sample_histogram


class Manipulator:
    def __init__(self, name, data):
        self.name = name
        self.JOINTS_OF_INTEREST = [
            SPINE_CHEST,
            CLAVICLE_RIGHT,
            SHOULDER_RIGHT,
            ELBOW_RIGHT,
            WRIST_RIGHT,
        ]
        self.picture_data = []
        self.tf_data = []
        self.theta_prob = {}
        self.trans_data = {}
        self.calibrate(data)
        self.t = 0
        self.theta = {
            index: self.tf_data[0][index][1]
            for index in INDICES_TO_STRINGS.keys()
        }

    def set(self, theta):
        for key, value in theta.items():
            self.theta[key] = value

    def render(self):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        origin = Joint()
        parents_index = [PELVIS]
        parents_tf = [compute_transform(
            origin, self.tf_data[0][PELVIS][0], self.theta[PELVIS])]
        while len(parents_index) != 0:
            next_parents_index = []
            next_parents_tf = []
            for parent_tf, parent_index in zip(parents_tf, parents_index):
                if parent_index in CHILD_INDICES.keys():
                    for child_index in CHILD_INDICES[parent_index]:
                        trans = self.trans_data[child_index]
                        euler = self.theta[child_index]
                        parent_p = parent_tf[:3, 3]
                        parent = transform_to_joint(parent_tf)
                        child_tf = compute_transform(parent, trans, euler)
                        child_p = child_tf[:3, 3]

                        ax.scatter(child_p[0], child_p[1], child_p[2])
                        ax.scatter(parent_p[0], parent_p[1], parent_p[2])
                        ax.plot(
                            [parent_p[0], child_p[0]],
                            [parent_p[1], child_p[1]],
                            [parent_p[2], child_p[2]],
                        )
                        if child_index in PARENT_INDICES.keys():
                            parents_index.append(child_index)
                            parents_tf.append(child_tf)

            parents_index = next_parents_index
            parents_tf = next_parents_tf

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        set_axes_equal(ax)
        ax.invert_zaxis()
        ax.view_init(azim=270, elev=105, )
        plt.tight_layout()
        plt.show()

    def calibrate(self, data):
        theta_instances = np.zeros((len(data), 32, 3))
        trans_instances = np.zeros((len(data), 32, 3))
        for idx, instance in enumerate(data):
            picture = {}
            for point in instance:
                p = point[2]
                q = point[3]
                index = point[0]
                picture[index] = Joint(p, q, index)
            self.picture_data.append(picture)

            tf = {}
            for index in INDICES_TO_STRINGS.keys():
                child_index = index
                child = picture[child_index]

                if child_index in PARENT_INDICES.keys():
                    parent_index = PARENT_INDICES[child_index]
                    parent = picture[parent_index]
                else:
                    parent = Joint()

                euler = compute_rotation(parent, child)
                trans = compute_translation(parent, child)
                child_tf = compute_transform(parent, trans, euler)
                np.testing.assert_array_almost_equal(child.t, child_tf)

                tf[index] = [trans, euler]
                theta_instances[idx, index] = np.array([
                    euler.z, euler.y, euler.x])
                trans_instances[idx, index] = trans[:-1]
            self.tf_data.append(tf)

        for index in INDICES_TO_STRINGS.keys():
            euler_z = theta_instances[:, index, 0]
            hist_z, bins_z = np.histogram(euler_z)
            euler_y = theta_instances[:, index, 1]
            hist_y, bins_y = np.histogram(euler_y)
            euler_x = theta_instances[:, index, 2]
            hist_x, bins_x = np.histogram(euler_x)
            prob = {
                'z': [hist_z, bins_z],
                'y': [hist_y, bins_y],
                'x': [hist_z, bins_x],
            }
            self.theta_prob[index] = prob
            self.trans_data[index] = \
                trans_instances[:, index, :].mean(axis=0)

    def sample(self, sample_joints=INDICES_TO_STRINGS.keys(), negate=True):
        theta = {}
        for index in sample_joints:
            hist_z, bins_z = self.theta_prob[index]['z']
            z = sample_histogram(hist_z, bins_z, negate=negate)
            hist_y, bins_y = self.theta_prob[index]['y']
            y = sample_histogram(hist_y, bins_y, negate=negate)
            hist_x, bins_x = self.theta_prob[index]['x']
            x = sample_histogram(hist_x, bins_x, negate=negate)
            theta[index] = Euler(z=z, y=y, x=x)
        return theta

    def forward_kinematic(self):
        raise NotImplementedError

    def backward_kinematic(self, gst):
        raise NotImplementedError

    def plan(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def compensate(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def _dxdy(self):
        raise NotImplementedError
