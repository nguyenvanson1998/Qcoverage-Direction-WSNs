import numpy as np

from ableplaceQcover.building_candidate import greedy_sensors
from nonplaceQcover.unit import Target


def read_file(path):
    with open(path) as f:
        targets = []

        theta = float(f.readline()) * np.pi / 180
        radius = float(f.readline())
        f.readline()

        n = int(f.readline())
        m = int(f.readline())

        for i in range(n):
            x, y, a = f.readline().split(',')
            target = Target(float(x), float(y), int(a), i)
            targets.append(target)

        return targets, m, theta, radius


if __name__ == '__main__':
    targets, m, theta, R = read_file('../test.txt')
    positions, angles = greedy_sensors(targets, m, theta, R)
    for pos, alpha in zip(positions, angles):
        print(pos, alpha)
