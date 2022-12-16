from nonplaceQcover.ga import GA
from nonplaceQcover.problem import Problem
from nonplaceQcover.unit import Target, Sensor


def read_file(path):
    with open(path) as f:
        sensors = []
        targets = []

        theta = float(f.readline())
        radius = float(f.readline())
        f.readline()

        n = int(f.readline())
        m = int(f.readline())

        for _ in range(n):
            x, y, a = f.readline().split(',')
            target = Target(float(x), float(y), int(a))
            targets.append(target)

        for _ in range(m):
            x, y = f.readline().split()
            sensor = Sensor(float(x), float(y), theta, radius)
            sensors.append(sensor)

        return Problem(sensors, targets)


if __name__ == '__main__':
    problem = read_file('../test.txt')
    solver = GA(problem)
    solution = solver.run()

    print('----------------')
    print('Results:')
    print('\tActivation vector:', solution.active)
    print('\tSensor angles:', solution.alpha)
    print('Metrics:')
    print('\tCQ =', problem.CQ(solution))
    print('\tQBI =', problem.QBI(solution))
    print('\tNumber of active sensors =', problem.active_sensor_count(solution))
