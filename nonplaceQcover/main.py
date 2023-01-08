import os
import sys
sys.path.append(os.getcwd())
from nonplaceQcover.ga import GA
from nonplaceQcover.problem import Problem
from nonplaceQcover.unit import Target, Sensor
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

        for i in range(n):
            x, y, a = f.readline().split(',')
            target = Target(float(x), float(y), int(a), i)
            targets.append(target)

        for _ in range(m):
            x, y = f.readline().split(',')
            sensor = Sensor(float(x), float(y), theta, radius)
            sensors.append(sensor)

        return Problem(sensors, targets)

def solve(inpath):
    
    outpath = inpath.replace('data', 'results')
    outpath = outpath.replace('.inp', '.out')
    #print(f"Doing Test {outpath}")
    problem = read_file(inpath)
    solver = GA(problem)
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