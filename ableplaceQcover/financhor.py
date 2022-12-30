from math import acos, cos, sin
from typing import Tuple

from utils import check_inside
import math
import numpy as np
import cmath
eps = np.finfo(float).eps



def get_points_inside(index, R, points, dis, keep_number):
    results = []

    keep_sensors = []

    for i in range(len(points)):
        if (i != index) and (dis[index][i] <= 2*R + eps):
            B = acos(dis[index][i]/(2*R))
            A = cmath.phase(complex(points[i][0] - points[index][0], points[i][1] - points[index][1] ))
            alpha =  A - B
            beta = A + B
            results.append((alpha, True))
            results.append((beta,False))
    
    results.sort(key=lambda x: (x[0], -x[1]))
    if len(results) == 0:
        return tuple([points[index][0], points[index][1], 1])
    else:
        count = 1
        res = np.empty(keep_number, dtype=object)
        
        for angle in results:
            if(angle[1] == True):
                count = count +1
            else:
                count = count -1
            best_result = [points[index][0] + R*cos(angle[0]), points[index][1] + R*sin(angle[0]), count]
            keep_sensors.append(best_result)

            # if count > res:
            #     res = count
            #     best_result = [points[index][0] + R*cos(angle[0]), points[index][1] + R*sin(angle[0])]
        keep_sensors.sort(key=lambda x: -x[2])
        return keep_sensors[:keep_number]



