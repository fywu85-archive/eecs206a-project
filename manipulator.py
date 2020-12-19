from kinematic import prod_exp
from macros import *
import numpy as np
from pytransform3d.rotations import matrix_from_quaternion
from pytransform3d.rotations import euler_zyx_from_matrix
from scipy.linalg import expm


class Joint:
    def __init__(self, p=None, q=None, index=None):
        self.index = index
        if p is not None:
            self.p = np.ones((4, ))
            self.p[:3] = np.array(p) / 1000
            self.q = q
            quat_rotation = matrix_from_quaternion(self.q)
            y180 = expm(np.array([
                [0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0],
            ]) * np.pi)
            self.r = y180.dot(quat_rotation)
            self.e = euler_zyx_from_matrix(self.r)
            self.t = np.zeros((4, 4))
            self.t[:3, :3] = self.r
            self.t[:, 3] = self.p
            self.t_inv = np.linalg.inv(self.t)
        else:
            self.p = None
            self.q = None
            self.r = None
            self.e = None
            self.t = None
            self.t_inv = None

    def update(self, p, q):
        self.p = np.ones((4,))
        self.p[:3] = np.array(p) / 1000
        self.q = q
        quat_rotation = matrix_from_quaternion(self.q.as_quat())
        y180 = expm(np.array([
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0],
        ]) * np.pi)
        self.r = y180.dot(quat_rotation)
        self.e = euler_zyx_from_matrix(self.r)
        self.t = np.zeros((4, 4))
        self.t[:3, :3] = self.r
        self.t[:, 3] = self.p
        self.t_inv = np.linalg.inv(self.t)


class Manipulator:
    def __init__(self, name, data):
        self.name = name
        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.l4 = None
        self.S = Joint()
        self.A = Joint()
        self.B = Joint()
        self.C = Joint()
        self.T = Joint()
        self.t = 0
        self.gsx = None
        self.xis = None
        self.calibrate(data)

    def step(self):
        raise NotImplementedError

    def forward_kinematic(self, theta):
        gs1 = np.matmul(prod_exp(self.xis[:, :3], theta[:3]), self.gsx[1])
        gs2 = np.matmul(prod_exp(self.xis[:, :6], theta[:6]), self.gsx[2])
        gs3 = np.matmul(prod_exp(self.xis[:, :9], theta[:9]), self.gsx[3])
        gst0 = np.matmul(prod_exp(self.xis, theta), self.gsx[4])
        gst1 = np.matmul(prod_exp(self.xis, theta), self.gsx[5])
        gst2 = np.matmul(prod_exp(self.xis, theta), self.gsx[6])
        gst3 = np.matmul(prod_exp(self.xis, theta), self.gsx[7])
        return [self.gsx[0], gs1, gs2, gs3, gst0, gst1, gst2, gst3]

    def backward_kinematic(self, gst):
        raise NotImplementedError

    def calibrate(self, data):
        gs0, gs1, gs2, gs3, gst0, gst1, gst2, gst3 = np.empty(8)
        xi0x, xi0y, xi0z, xi1x, xi1y, xi1z, \
            xi2x, xi2y, xi2z, xi3x, xi3y, xi3z = np.empty(12)
        for point in data[0]:
            joint = Joint(point[2], point[3], point[0])
            if joint.index == SPINE_CHEST:
                gs0 = self._compute_g(joint)
                xi0x, xi0y, xi0z = self._compute_xi(joint)
            elif joint.index == CLAVICLE_RIGHT:
                gs1 = self._compute_g(joint)
                xi1x, xi1y, xi1z = self._compute_xi(joint)
            elif joint.index == SHOULDER_RIGHT:
                gs2 = self._compute_g(joint)
                xi2x, xi2y, xi2z = self._compute_xi(joint)
            elif joint.index == ELBOW_RIGHT:
                gs3 = self._compute_g(joint)
                xi3x, xi3y, xi3z = self._compute_xi(joint)
            elif joint.index == WRIST_RIGHT:
                gst0 = self._compute_g(joint)
            elif joint.index == HAND_RIGHT:
                gst1 = self._compute_g(joint)
            elif joint.index == HANDTIP_RIGHT:
                gst2 = self._compute_g(joint)
            elif joint.index == THUMB_RIGHT:
                gst3 = self._compute_g(joint)
        self.gsx = np.array([gs0, gs1, gs2, gs3, gst0, gst1, gst2, gst3])
        self.xis = np.array([xi0x, xi0y, xi0z, xi1x, xi1y, xi1z,
                             xi2x, xi2y, xi2z, xi3x, xi3y, xi3z]).T

        arm_lengths = []
        for frame in data:
            skeleton = {}
            for point in frame:
                joint = Joint(point[2], point[3], point[0])
                if joint.index in JOINTS_OF_INTEREST:
                    skeleton[joint.index] = joint
            lengths = []
            for child in JOINTS_OF_INTEREST[1:]:
                parent = SKELETON_INDICES[child]
                p1 = skeleton[child].p
                p2 = skeleton[parent].p
                length = np.sqrt(
                    (p1[0] - p2[0])**2 +
                    (p1[1] - p2[1])**2 +
                    (p1[2] - p2[2])**2)
                lengths.append(length)
            arm_lengths.append(lengths)
        arm_lengths = np.array(arm_lengths)
        self.l1, self.l2, self.l3, self.l4 = arm_lengths.mean(axis=0)
        print(arm_lengths.mean(axis=0))
        print(arm_lengths.std(axis=0))
        print('Calibration completed.')

    @staticmethod
    def _compute_g(joint):
        return np.array([
            joint.r[0, :].tolist() + [joint.p[0]],
            joint.r[1, :].tolist() + [joint.p[1]],
            joint.r[2, :].tolist() + [joint.p[2]],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    @staticmethod
    def _compute_xi(joint):
        p = list(joint.p)
        wx = list(joint.r[:, 0])
        xix = list(-np.cross(wx, p)) + wx
        wy = list(joint.r[:, 1])
        xiy = list(-np.cross(wy, p)) + wy
        wz = list(joint.r[:, 2])
        xiz = list(-np.cross(wz, p)) + wz
        return xix, xiy, xiz

    def plan(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def compensate(self):
        raise NotImplementedError

    def _dxdy(self):
        raise NotImplementedError
