from ableplaceQcover.unit import Target
from ableplaceQcover.unit import Sensor
from financhor import get_points_inside
def select_sroll_points(target: Target, r: float):
    list_rool = []
    list_rool.append([target.pos[0], target.pos[1]])
    # first case
    list_rool.append([target.pos[0] - r, target.pos[1]])
    list_rool.append([target.pos[0] + r, target.pos[1]])
    list_rool.append([target.pos[0], target.pos[1] + r])
    list_rool.append([target.pos[0], target.pos[1] -r])
    return list_rool



def get_sensor_candidate(targets, r):
    candidates = []
    candidates.extend([[targets[i].pos[0], targets[i].pos[1]] for i in range(len(targets))] )
    for t in targets:
        keep_number = t.k_cover
        list_rool = select_sroll_points(t, r)
        keeps = []
        for rool in list_rool:
            x = get_points_inside(rool,targets,r,keep_number)
            keeps.extend(x)
        keeps.sort(key=lambda x: -x[2])
        if len(keeps) > keep_number:
            keeps = keeps[:int(keep_number)]
        candidates.extend(keeps)
    return candidates

def cvert_cand_to_sensor(candidates, theta, radius):
    """
    This function is convert the list of point to sensor class
    """
    return [Sensor(float(candidates[i][0]), float(candidates[i][1]), theta=theta, radius=radius) for i in range(len(candidates))]



