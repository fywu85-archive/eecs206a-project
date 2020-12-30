from macros import *
from manipulator import Manipulator
from utils import parse_data


if __name__ == '__main__':
    data = parse_data('data/data_random.txt')
    arm = Manipulator('ARM R', data)
    # arm.render()

    sample_joints = [SPINE_CHEST, CLAVICLE_RIGHT, SHOULDER_RIGHT]
    theta_rand = arm.sample(sample_joints=sample_joints)
    arm.set(theta_rand)
    arm.render()
