import os
import sys
sys.path.append(os.getcwd())
from nonplaceQcover.ga import GA
from nonplaceQcover.problem import Problem
from nonplaceQcover.unit import Target, Sensor
from ableplaceQcover.building_candidate import get_sensor_candidate, cvert_cand_to_sensor
import numpy as np

def read_file(path):
    with open(path) as f:
        sensors = []
        targets = []

        theta = float(f.readline()) * np.pi / 180
        radius = float(f.readline())
        f.readline()

        n = int(f.readline())
        m = int(f.readline())

        for _ in range(n):
            x, y, a = f.readline().split(',')
            target = Target(float(x), float(y), int(a))
            targets.append(target)
        candidates = get_sensor_candidate(targets=targets,r = radius)
        sensors = cvert_cand_to_sensor(candidates=candidates,theta=theta,radius=radius)
        print(len(sensors))
        

        return Problem(sensors, targets)


if __name__ == '__main__':
    problem = read_file('./data/test_able.txt')
    # solver = GA(problem)
    # solution = solver.run()

    # print('----------------')
    # print('Results:')
    # print('\tActivation vector:', solution.active)
    # print('\tSensor angles:', solution.alpha)
    # print('Metrics:')
    # print('\tCQ =', problem.CQ(solution))
    # print('\tQBI =', problem.QBI(solution))
    # print('\tNumber of active sensors =', problem.active_sensor_count(solution))
