import numpy as np

from ableplaceQcover.financhor import get_points_inside
from nonplaceQcover.unit import Sensor


def find_angle(sensor: Sensor, targets: list):
    # list of cover-able targets
    tg = [target for target in targets
          if np.sqrt(np.sum(np.square(sensor.pos - target.pos))) <= sensor.radius]
    if len(tg) == 0:
        return None

    # sort targets by angle formed with sensor
    all_angles = []
    for target in targets:
        v_vector = target.pos - sensor.pos
        angle = np.arctan2(v_vector[1], v_vector[0]) + sensor.theta / 2
        all_angles.append(angle)

    indices = np.argsort(all_angles)
    tg = np.array(tg)[indices]
    all_angles = np.array(all_angles)[indices]

    # slide through potential angles while updating number of covered targets
    best_cover = [0, 0]
    best_angle = None
    st = 0
    fn = 0
    for angle in all_angles:
        while st < len(tg) and not sensor.cover(angle, tg[st]):
            st += 1
        while fn < len(tg) and sensor.cover(angle, tg[fn]):
            fn += 1
        if st >= fn:
            break
        if fn - st > best_cover[1] - best_cover[0]:
            best_angle = angle
            best_cover = [st, fn]

    return best_angle, tg[best_cover[0]:best_cover[1]]


def greedy_sensors(targets: list, m: int, theta: float, R: float):
    # prepare necessary info
    n = len(targets)
    points = np.empty((n, 2), dtype=float)
    for i, target in enumerate(targets):
        points[i] = target.pos

    covered = np.zeros(n, dtype=int)
    dis = np.zeros([n, n], dtype=float)

    for i, t1 in enumerate(targets):
        for j, t2 in enumerate(targets):
            dis[i, j] = np.sqrt(np.sum(np.square(t1.pos - t2.pos)))

    positions = []  # candidate sensors' positions
    angles = []     # candidate sensors' angles

    def get_priority(idx):
        req =targets[idx].k_cover - covered[idx]
        return req if req > 0 else req + m**2

    # find m sensors greedily
    while len(positions) < m:
        best_idx = np.argmax([get_priority(idx) for idx in range(len(targets))])
        vip_target = targets[best_idx]

        sensor = get_points_inside(vip_target.index, R, points, dis, 1)[0]
        sensor = Sensor(sensor[0], sensor[1], theta, R)
        positions.append(sensor.pos)

        angle, covered_targets = find_angle(sensor, targets)
        angles.append(angle)
        for tg in covered_targets:
            covered[tg.index] += 1

    return positions, angles
