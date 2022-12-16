from typing import List

import numpy as np

from nonplaceQcover.individual import Individual
from nonplaceQcover.unit import Sensor, Target


class Problem:
    def __init__(self, sensors: List[Sensor], targets: List[Target]):
        self.sensors = sensors
        self.targets = targets

    def get_achieved_coverage(self, p: Individual):
        phi = np.zeros(len(self.targets), dtype=int)
        for i, sensor in enumerate(self.sensors):
            if p.active[i]:
                for j, target in enumerate(self.targets):
                    if sensor.cover(p.alpha[i], target):
                        phi[j] += 1
        return phi

    def active_sensor_count(self, p: Individual):
        return np.count_nonzero(p.active)

    def QBI(self, p: Individual):
        phi = self.get_achieved_coverage(p)
        K = np.array([target.k_cover for target in self.targets])
        if np.sum(phi) == 0:
            return 0
        QBI = (np.sum(phi) ** 3) / np.sum(np.square(phi)) \
              * np.sum(np.square(K)) / (np.sum(K) ** 3)
        return QBI

    def CQ(self, p: Individual):
        cq = 0
        for i, sensor in enumerate(self.sensors):
            if p.active[i]:
                for j, target in enumerate(self.targets):
                    if sensor.cover(p.alpha[i], target):
                        distance = np.sqrt(np.sum(np.square(target.pos - sensor.pos)))
                        cq += 1 - (distance / sensor.radius) ** 2
        return cq

    def Pc(self, p: Individual):
        Pa = 5.268
        Pi = 1.473
        Ps = 0.058

    def evaluate(self, p: Individual):
        phi = self.get_achieved_coverage(p)
        if all([phi[j] >= target.k_cover for j, target in enumerate(self.targets)]):
            return self.active_sensor_count(p)
        return self.QBI(p)
