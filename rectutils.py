__author__ = 'sz372'


import cv2
import numpy as np
import utils


def on_same_line(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if abs(y1 - y2) > min(h1,h2)/float(4):
        return False
    return True


def next_on_same_line(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if not on_same_line(r1, r2) or abs(x1 + w1 - x2) > min(h1,h2)/float(2):
        return False
    return True


def on_consecutive_line(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if abs(y1 + h1 - y2) > min(h1,h2)/float(2):
        return False
    return True


def same_height(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if abs(h1 - h2) > min(h1,h2)/float(5):
        return False
    return True


def filter_duplicates(rects):
    print "Filtering %d regions..." % (len(rects))

    C = np.zeros((len(rects), len(rects)), dtype=bool)
    for i, r1 in enumerate(rects):
        for j, r2 in enumerate(rects):
            if abs(r1[0] - r2[0]) < 10 and abs(r1[1] - r2[1]) < 10 and abs(r1[2] - r2[2]) < 10 and abs(r1[3] - r2[3]) < 10:
                C[i, j] = True
    rects =  __bfs_bbx(rects, C)

    print "\tto %d regions" % (len(rects))
    return rects


def combine_inside(rects):
    print "Combining %d regions..." % (len(rects))

    C = np.zeros((len(rects), len(rects)), dtype=bool)
    for i, r1 in enumerate(rects):
        for j, r2 in enumerate(rects):
            if (r1[0] <= r2[0] <= r1[0] + r1[2] and r1[0] <= r2[0] + r2[2] <= r1[0] + r1[2]
                and r1[1] <= r2[1] <= r1[1] + r1[3] and r1[1] <= r2[1] + r2[3] <= r1[1] + r1[3]) or (
                                        r2[0] <= r1[0] <= r2[0] + r2[2] and r2[0] <= r1[0] + r1[2] <= r2[0] + r2[2]
                and r2[1] <= r1[1] <= r2[1] + r2[3] and r2[1] <= r1[1] + r1[3] <= r2[1] + r2[3]):
                C[i, j] = True
    rects =  __bfs_bbx(rects, C)

    print "\tto %d regions" % (len(rects))
    return rects



def find_words(rects):
    C = np.zeros((len(rects), len(rects)), dtype=bool)
    for i, r1 in enumerate(rects):
        x1, y1, w1, h1 = r1
        for j, r2 in enumerate(rects):
            x2, y2, w2, h2, = r2
            if i == j or (abs(y1 - y2) < min(h1,h2)/float(2) and (abs(x1 + w1 - x2) < 10 or abs(x2 + w2 - x1) < 10)):
                C[i, j] = True

    return __bfs_bbx(rects, C)


def __bfs_bbx(rects, C):
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

        j = 0
        while j < len(neighbors):
            s = neighbors[j]
            s_neighbors = set(np.where(C[s] == True)[0])
            s_neighbors = s_neighbors - visited
            neighbors.extend(s_neighbors)
            visited = visited.union(s_neighbors)
            j += 1

    newrects = []
    for value in isclose.values():
        newrects.append(cv2.boundingRect(np.concatenate([utils.points(rects[r]) for r in value])))

    return newrects
