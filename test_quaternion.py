from macros import *
from utils import Joint
from utils import \
    parse_data, compute_rotation, compute_translation, compute_transform


if __name__ == '__main__':
    data = parse_data('data/data_calibrate.txt')[0]

    elbow = None
    wrist = None
    for point in data:
        idx = int(point[0])
        if idx == ELBOW_RIGHT:
            elbow = Joint(point[2], point[3], point[0])
        elif idx == WRIST_RIGHT:
            wrist = Joint(point[2], point[3], point[0])

    euler = compute_rotation(elbow, wrist)
    trans = compute_translation(elbow, wrist)
    tf = compute_transform(elbow, trans, euler)

    import numpy as np
    np.testing.assert_array_almost_equal(wrist.t, tf)
    print('Passed')
