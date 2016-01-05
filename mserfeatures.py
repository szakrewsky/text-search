__author__ = 'sz372'


import cv2
import numpy as np
import utils


def _filter(rects):
    C = np.zeros((len(rects), len(rects)), dtype=bool)
    for i, r1 in enumerate(rects):
        for j, r2 in enumerate(rects):
            if abs(r1[0] - r2[0]) < 10 and abs(r1[1] - r2[1]) < 10 and abs(r1[2] - r2[2]) < 10 and abs(r1[3] - r2[3]) < 15:
                C[i, j] = True

    visited = set()
    isclose = {}
    for i in range(0, len(rects)):
        if i in visited:
            continue

        visited.add(i)
        neighbors = isclose.get(i, [])
        neighbors.extend(np.where(C[i] == True)[0])
        isclose[i] = neighbors
        visited = visited | set(neighbors)

        for j in range(0, len(neighbors)):
            s = neighbors[j]
            s_neighbors = set(np.where(C[s] == True)[0])
            s_neighbors = s_neighbors - visited
            neighbors.extend(s_neighbors)
            visited.union(s_neighbors)

    newrects = []
    for key, value in isclose.items():
        if value:
            newrects.append(cv2.boundingRect(np.concatenate([utils.points(rects[r]) for r in value])))
        else:
            newrects.append(rects[key])

    return newrects


def get_features(img_grey):
    mser = cv2.MSER_create()
    points = mser.detectRegions(img_grey, None)
    rects = map(cv2.boundingRect, points)
    print "Found %d MSER regions" % (len(rects))
    rects = _filter(rects)
    print "Filtered to %d MSER regions" % (len(rects))
    return rects