__author__ = 'sz372'


import cv2
import numpy as np
from rectutils import filter_duplicates, combine_inside


def get_features(img_grey):
    mser = cv2.MSER_create()
    regions = mser.detectRegions(img_grey, None)
    rects = map(cv2.boundingRect, regions)
    return filter_duplicates(rects)


def get_inverse_features_with_canny(img, regions):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.blur(img_grey, (11,11))
    mask = cv2.Canny(img_grey, 100, 200, apertureSize=5)
    for r in regions:
        x, y, w, h = r
        mask[y:y+h,x:x+w] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3)))

    # cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
    # cv2.imshow('tmp', mask)
    # cv2.waitKey()

    contours = _compute_outer_contours(mask)
    print "Found %d contours" % (len(contours),)
    rects = map(cv2.boundingRect, contours)
    crects = [(c,r) for c,r in zip(contours, rects) if _get_size(*img_grey.shape, p=0.01) < r[2]*r[3] < _get_size(*img_grey.shape, p=0.20)]
    crects = [(c,r) for c,r in crects if 5 > float(r[2])/r[3] > (1.0/5)]
    print "Found %d contours" % (len(crects),)

    #contours = [c for c,r in crects]
    #img = cv2.drawContours(img, contours, -1, (0,0,255), 2)

    rects = [r for c,r in crects]
    return combine_inside(filter_duplicates(rects))


def get_inverse_features(img, regions):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.equalizeHist(img_grey)

    contours = []
    for i in range(0, 256, 25):
        ret, img_bin = cv2.threshold(img_grey, i, 255, cv2.THRESH_BINARY)
        ret, img_bin_inv = cv2.threshold(img_grey, i, 255, cv2.THRESH_BINARY_INV)

        contours = np.concatenate([contours, _get_masked_contours(img_bin, img_bin_inv.copy(), regions)])

    print "Found %d contours" % (len(contours),)
    contours = [c for c in contours if _get_size(*img_grey.shape, p=0.0025) < cv2.contourArea(c) < _get_size(*img_grey.shape, p=0.25)]
    rects = map(cv2.boundingRect, contours)
    crects = [(c,r) for c,r in zip(contours, rects) if r[2]*r[3] < _get_size(*img_grey.shape, p=0.25)]# and cv2.contourArea(c)/(r[2]*r[3]) > 0.25]
    crects = [(c,r) for c,r in crects if 5 > float(r[2])/r[3] > (1.0/5) ]
    print "Found %d contours" % (len(crects),)

    #contours = [c for c,r in crects]
    #img = cv2.drawContours(img, contours, -1, (0,0,255), 2)

    rects = [r for c,r in crects]
    return combine_inside(filter_duplicates(rects))


def _get_masked_contours(img_bin, img_bin_inv, regions):
    for r in regions:
        x, y, w, h = r
        img_bin[y:y+h,x:x+w] = 0
        img_bin_inv[y:y+h,x:x+w] = 0
    contours = _compute_outer_contours(img_bin)
    contours_inv = _compute_outer_contours(img_bin_inv)
    return np.concatenate([contours, contours_inv])


def _compute_outer_contours(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # why?
    if hierarchy is None:
        return []
    mask = np.where(hierarchy[0][:,3] < 0)[0]
    tmp = np.empty((len(contours),), dtype=object)
    tmp[:] = contours
    return tmp[mask]


def _get_size(n1, n2, p):
    return int((n1 * n2) * p)

