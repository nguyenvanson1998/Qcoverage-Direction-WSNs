import os
import sys
from multiprocessing import Pool

sys.path.append(os.getcwd())
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


def solve(inpath):
    outpath = inpath.replace('test', 'results')
    outpath = outpath.replace('.inp', '.out')

    targets, m, theta, R = read_file(inpath)
    problem, solution = greedy_sensors(targets, m, theta, R)

    with open(outpath, 'w+') as fout:
        fout.writelines(f'\tSensor x-axes: {[s.pos[0] for s in problem.sensors]}\n')
        fout.writelines(f'\tSensor y-axes: {[s.pos[1] for s in problem.sensors]}\n')
        fout.writelines(f'\tSensor angles: {solution.alpha}\n')
        fout.writelines('Metrics:\n')
        fout.writelines(f'\tCQ = {problem.CQ(solution)}\n')
        fout.writelines(f'\tQBI = {problem.QBI(solution)}\n')
        fout.writelines(f'\tNumber of active sensors = {problem.active_sensor_count(solution)}\n')
        fout.writelines(f'\tDI = {problem.DI(solution)}\n')
        fout.writelines(f'\tPower Consumming = {problem.Pc(solution)}\n')
    print(f"Done Test {outpath}")


if __name__ == '__main__':
    root = './test/'
    data = os.listdir(root)
    inpath = [root + i for i in data]
    with Pool(2) as p:
        p.map(solve, inpath)
