#!/bin/python

"""
USAGE:
    text-search.py <string> <image>...
"""


from classifier import load_classifier
import cv2
import docopt
from features import hogdesc
import itertools
import numpy as np
import ocr
import searchcriteria
from searchcriteria import SearchCriteria
import templates
import textlines
import time


def prepare_detect(img, rects):
    vec = np.empty((len(rects), 24, 24), np.uint8)
    for i in range(0,len(rects)):
        x, y, w, h = rects[i]
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi, (24,24))
        vec[i] = roi
    return hogdesc(vec)


def draw_rects(winname, img, rects, color):
    for rect in rects:
        x, y, w, h = rect
        img = cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)


if __name__ == '__main__':

    arguments = docopt.docopt(__doc__)

    sc = SearchCriteria.parse(arguments['<string>'])

    classifier = load_classifier()

    for i in arguments['<image>']:
        img = cv2.imread(i)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(img_grey, (-1,-1), 3)
        img_grey = cv2.addWeighted(img_grey, 1.5, blur, -0.5, 0)

        mser = cv2.MSER_create()
        points = mser.detectRegions(img_grey, None)
        rects = map(cv2.boundingRect,points)
        # filter aspect ratio
        rects = [r for r in rects if 0.5 < float(r[2])/float(r[3]) < 2]

        start = time.time()
        print "Running classifier..."
        vec = prepare_detect(img_grey, rects)
        result = classifier.predict(vec)
        print "took %ds" % (time.time() - start,)

        result = result.reshape(-1)
        matches = np.in1d(result, list(sc.indexset()))
        rects = np.array(rects, int)[matches]
        result = result[matches]

        charmap = {}
        for c, rect in zip(result, rects):
            key = searchcriteria.get_label_value(int(c)).lower()
            rectarray = charmap.get(key, [])
            rectarray.append(rect)
            # TODO needed?
            charmap[key] = rectarray

        # Filter duplicates
        for key in charmap:
            rectarray = charmap[key]
            line = textlines.Line(rectarray, [key]*len(rectarray))
            line.filter()
            charmap[key] = line.components

        templates2d = templates.get_templates(sc.tokens)
        for key in charmap:
            temparray = []
            rectarray = charmap.get(key)
            candidates = []
            if key in templates2d:
                candidates.extend(templates2d[key])
            if key in searchcriteria._NUMBERS and '\d' in templates2d:
                candidates.extend(templates2d['\d'])
            if len(candidates) == 0:
                raise KeyError("No templates for key " + key)

            for r in rectarray:
                for c in candidates:
                        temparray.append(c.match2d(r))
            charmap[key] = temparray

        temparray = list(itertools.chain(*charmap.itervalues()))

        start = time.time()
        print "Running OCR..."
        matches = []
        for r in temparray:
            if r[0] < 0 or r[1] < 0 or r[2] > img.shape[1] or r[3] > img.shape[0]:
                continue

            roi = img_grey[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
            text = ocr.ocr(roi)
            for m in templates.match(sc.tokens, text.lower()):
                matches.append(r)

        print "took %ds" % (time.time() - start,)
        draw_rects('Matches ' + i, img.copy(), matches, (0,0,255))
    cv2.waitKey()