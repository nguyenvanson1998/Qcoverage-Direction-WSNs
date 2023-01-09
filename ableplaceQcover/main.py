import os
import sys
sys.path.append(os.getcwd())
from ableplaceQcover.ga import GA
from ableplaceQcover.problem import Problem
from ableplaceQcover.unit import Target, Sensor
from ableplaceQcover.building_candidate import get_sensor_candidate, cvert_cand_to_sensor
import numpy as np
from multiprocessing import Pool

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
            target = Target(float(x), float(y), float(a))
            targets.append(target)
        candidates = get_sensor_candidate(targets=targets,r = radius)
        sensors = cvert_cand_to_sensor(candidates=candidates,theta=theta,radius=radius)
        print(len(sensors))
        

        return Problem(sensors, targets,m)


# if __name__ == '__main__':
#     problem = read_file('./data/test_gr1_180sensor.inp')
#     solver = GA(problem, './data/test_gr1_180sensor.inp')
#     solution = solver.run()

#     print('----------------')
#     print('Results:')
#     print('\tActivation vector:', solution.active)
#     print('\tSensor angles:', solution.alpha)
#     print('Metrics:')
#     print('\tCQ =', problem.CQ(solution))
#     print('\tQBI =', problem.QBI(solution))
#     print('\tNumber of active sensors =', problem.active_sensor_count(solution))


def solve(inpath):
    
    outpath = inpath.replace('test', 'results')
    outpath = outpath.replace('.inp', '.out')
    #print(f"Doing Test {outpath}")
    problem = read_file(inpath)
    solver = GA(problem, outpath)
    solution = solver.run()
    with open(outpath, 'w+') as fout:
        fout.writelines(f'\tActivation vector:{solution.active}\n')
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
        p.map(solve,inpath)