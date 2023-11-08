import os
import sys
sys.path.append(os.getcwd())
from nonplaceQcover.ga import GA
from nonplaceQcover.problem import Problem
from nonplaceQcover.unit import Target, Sensor
import numpy as np
from multiprocessing import Pool
import json

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
            target = Target(float(x), float(y), float(a), i)
            targets.append(target)

        for _ in range(m):
            x, y = f.readline().split(',')
            sensor = Sensor(float(x), float(y), theta, radius)
            sensors.append(sensor)

        return Problem(sensors, targets)

def solve(inpath):
    
    outpath = inpath.replace('test', 'results')
    outpath = outpath.replace('.inp', '.json')
    #print(f"Doing Test {outpath}")
    problem = read_file(inpath)
    solver = GA(problem, outpath)
    solution = solver.run()
    data = {
    "Activation_vector": solution.active.tolist(),
    "Sensor_angles": solution.alpha.tolist(),
    "Metrics": {
        "CQ": problem.CQ(solution),
        "QBI": problem.QBI(solution),
        "No.Active": problem.active_sensor_count(solution),
        "DI": problem.DI(solution),
        "PC": problem.Pc(solution)
    }
    }
    with open(outpath, "w") as f:
        json.dump(data, f, indent=4)  # Thêm indent=4 để dễ đọc và định dạng JSON đẹp

    print(f"Done Test {outpath}")
    

if __name__ == '__main__':
    solve("test/test.inp")
    # root = './test/'
    # data = os.listdir(root)
    # inpath = [root + i for i in data]
    # with Pool(2) as p:
    #     p.map(solve,inpath)