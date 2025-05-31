import numpy as np


class Target:
    def __init__(self, x, y, k_cover, index):
        self.index = index
        self.pos = np.array([x, y])
        self.k_cover = k_cover


class Sensor:
    def __init__(self, x, y, theta, radius):
        self.pos = np.array([x, y])
        self.theta = theta
        self.radius = radius

    # check if this sensor cover the target with alpha as active angle
    def cover(self, alpha: float, target: Target):
        # angle constraint
        f_vector = np.array([np.cos(alpha), np.sin(alpha)])
        v_vector = target.pos - self.pos
        angle = np.arccos(np.dot(f_vector, v_vector)/(np.linalg.norm(f_vector)*(np.linalg.norm(v_vector))))

        # radius constraint
        distance = np.sqrt(np.sum(np.square(v_vector)))

        return angle <= self.theta / 2 and distance <= self.radius
