__author__ = 'sz372'

import cv2
import numpy as np

SZ=24
bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img


def hog2(img, n=4, B=16):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(B*ang/(2*np.pi))    # quantizing binvalues in (0...B)
    bin_cells = [bins[j:j+n,i:i+n] for j in range(0, img.shape[0], n) for i in range(0, img.shape[1], n)]
    mag_cells = [mag[j:j+n,i:i+n] for j in range(0, img.shape[0], n) for i in range(0, img.shape[1], n)]
    hists = [np.bincount(b.ravel(), m.ravel(), B) for b, m in zip(bin_cells, mag_cells)]
    hist = np.array(hists).reshape((img.shape[0]/n, img.shape[1]/n, B))
    return hist


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


def hogdesc(train_cells):
    deskewed = map(deskew,train_cells)
    hogdata = map(hog,deskewed)
    return np.float32(hogdata).reshape(-1,64)


def _char_obj_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, -1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    cimg, contours, hier = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.namedWindow('chraobj', cv2.WINDOW_NORMAL)
    # cv2.imshow('chraobj', img)
    # cv2.waitKey()
    return len(contours)


def char_obj_features(imgs):
    return np.float32(map(_char_obj_features, imgs)).reshape(-1,1)