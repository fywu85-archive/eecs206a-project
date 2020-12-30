from manipulator import Manipulator
from utils import parse_data


if __name__ == '__main__':
    data = parse_data('data/data_random.txt')
    arm = Manipulator('ARM R', data)
    # print(arm.theta_prob)
    # print(arm.trans_data)
    arm.render()

    # euler_init = arm.data[0]
    # arm.set(euler_init)
    # arm.render()
    #
    # euler_rand = arm.sample()
    # arm.set(euler_rand)
    # arm.render()
