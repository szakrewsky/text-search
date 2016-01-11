__author__ = 'sz372'


import cv2
from rectutils import filter_duplicates


def get_features(img_grey):
    mser = cv2.MSER_create()
    points = mser.detectRegions(img_grey, None)
    rects = map(cv2.boundingRect, points)
    return filter_duplicates(rects)
