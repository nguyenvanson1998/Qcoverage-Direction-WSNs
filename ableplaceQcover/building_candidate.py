import numpy as np

from ableplaceQcover.financhor import get_points_inside
from nonplaceQcover.individual import Individual
from nonplaceQcover.problem import Problem
from nonplaceQcover.unit import Sensor


def find_angle(sensor: Sensor, targets: list):
    # list of cover-able targets
    tg = [target for target in targets
          if np.sqrt(np.sum(np.square(sensor.pos - target.pos))) <= sensor.radius]
    if len(tg) == 0:
        return None

    # sort targets by angle formed with sensor
    all_angles = []
    for target in tg:
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
    # points = np.empty((n, 2), dtype=float)

    cover_requirement = np.array([tg.k_cover for tg in targets], dtype=int)
    covered = np.zeros(n, dtype=int)
    # dis = np.zeros([n, n], dtype=float)
    #
    # for i, t1 in enumerate(targets):
    #     for j, t2 in enumerate(targets):
    #         dis[i, j] = np.sqrt(np.sum(np.square(t1.pos - t2.pos)))

    sensors = []    # candidate sensors' positions
    angles = []     # candidate sensors' angles

    def get_priority(idx):
        req =targets[idx].k_cover - covered[idx]
        return req if req > 0 else req + m**2

    # find m sensors greedily
    while len(sensors) < m and np.any(covered < cover_requirement):
        best_idx = np.argmax([get_priority(idx) for idx in range(len(targets))])
        vip_target = targets[best_idx]

        sensor = get_points_inside(vip_target.pos, targets, R, 1)[0]
        sensor = Sensor(sensor[0], sensor[1], theta, R)
        sensors.append(sensor)

        angle, covered_targets = find_angle(sensor, targets)
        angles.append(angle)
        for tg in covered_targets:
            covered[tg.index] += 1

    problem = Problem(sensors, targets)
    individual = Individual(len(sensors))
    individual.active = np.full(individual.dim, True, dtype=bool)
    individual.alpha = np.array(angles)
    return problem, individual
