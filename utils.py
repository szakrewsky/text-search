__author__ = 'sz372'

import cv2
import itertools
import numpy as np


def intersects(rect1, rect2):
    def _intersects(a, b, dim):
        if dim == 'y':
            p = 1
            s = 3
        elif dim == 'x':
            p = 0
            s = 2
        else:
            raise ValueError("either x or y for dim")

        if a[p] > b[p]:
            t = a
            a = b
            b = t

        if a[p] + a[s] > b[p]:
            return True
        return False

    return _intersects(rect1, rect2, 'y') and _intersects(rect1, rect2, 'x')


def points(rect):
    x,y,w,h = rect
    return [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]


def combine_overlapping_rectangles(img, rects):
    def _combine_overlapping_rectangles(rects):
        used = []
        results = []
        overlap = False
        for i in range(0, len(rects)):
            if i in used:
                continue
            a = rects[i]
            for j in range(i+1,len(rects)):
                if j in used:
                    continue
                b = rects[j]
                if intersects(a, b):
                    overlap = True
                    x,y,w,h = cv2.boundingRect(np.array(points(a) + points(b)))
                    a = (x,y,w-1,h-1)
                    used.append(j)

            results.append(a)
        return overlap, results

    overlap, results = _combine_overlapping_rectangles(rects)
    while overlap:
        overlap, results = _combine_overlapping_rectangles(results)
    return results


def get_region_rects_for_groups(groups, regions):
    ret = []
    for group in groups:
        rects = np.empty((len(group), 4), dtype=int)
        for i in range(0, len(group)):
            channel, idx = group[i]
            rects[i] = regions[channel][idx].rect
        ret.append(rects)
    return ret


def get_region_rects_for_group(group, regions):
    rects = np.empty((len(group), 4), dtype=int)
    for i in range(0, len(group)):
        channel, idx = group[i]
        rects[i] = regions[channel][idx].rect
    return rects


def inliers(arr, n=1.25):
    return np.abs(arr - np.mean(arr)) < (n * np.std(arr))


def split_groups_on_horizontal_space(ctx, groups):
    new_groups = []
    new_rects = []
    regions = ctx.regions
    for group in groups:
        region_rects = get_region_rects_for_group(group, regions)
        region_rects = region_rects[np.argsort(region_rects[:,0])]
        whitespace = region_rects[1:,0] - (region_rects[:-1,0] + region_rects[:-1,2])
        indices = np.where(inliers(whitespace) == False)[0] + 1

        for new_group in np.split(region_rects, indices):
            new_rects.append(cv2.boundingRect(np.concatenate([points(r) for r in new_group])))
            new_groups.append(new_group)
    return new_groups, new_rects


def draw_rects(img, rects):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)