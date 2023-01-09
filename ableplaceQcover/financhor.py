from math import acos, cos, sin
from typing import Tuple

from utils import check_inside
import math
import numpy as np
import cmath
eps = np.finfo(float).eps


def distance(rool, point):
    dis = math.sqrt((rool[0] - point.pos[0])*(rool[0] - point.pos[0]) + (rool[1] - point.pos[1])*(rool[1] - point.pos[1]))
    return dis
# def get_points_inside(index, R, points, dis, keep_number):
def get_points_inside(rool, points, R, keep_number=1):
    n = len(points)
    dis = np.full(n, 1)
    for i in range(n):
        dis[i] = distance(rool, points[i])

    results = []

    keep_sensors = []
    count = 0 
    for i in range(len(points)):

        if dis[i] <= eps:
            count = count +1
        
        elif (dis[i] <= 2*R + eps) :
            B = acos(dis[i]/(2*R))
            A = cmath.phase(complex(points[i].pos[0] - rool[0], points[i].pos[1] - rool[1] ))
            alpha =  A - B
            beta = A + B
            results.append((alpha, True))
            results.append((beta,False))
    
    results.sort(key=lambda x: (x[0], -x[1]))
    if len(results) == 0:
        return []
    else:
        
        for angle in results:
            if(angle[1] == True):
                count = count +1
            else:
                count = count -1
            best_result = [rool[0] + R*cos(angle[0]), rool[1] + R*sin(angle[0]), count]
            keep_sensors.append(best_result)

            # if count > res:
            #     res = count
            #     best_result = [points[index][0] + R*cos(angle[0]), points[index][1] + R*sin(angle[0])]
        keep_sensors.sort(key=lambda x: -x[2])
        if len(keep_sensors) > keep_number:
            keep_sensors = keep_sensors[:int(keep_number)]
        for s in keep_sensors:
            if s[2] <1:
                keep_sensors.remove(s)
        return keep_sensors



