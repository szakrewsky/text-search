#!/bin/python

"""
USAGE:
    text-search.py <string> <image>...
"""


from classifier import load_classifier
import cv2
import docopt
from features import hogdesc
import numpy as np
import ocr
from random import randint
from rectutils import next_on_same_line, on_consecutive_line, same_height
import searchcriteria
from searchcriteria import SearchCriteria
import templates
import time
import utils


def prepare_detect(img, rects):
    vec = np.empty((len(rects), 24, 24), np.uint8)
    for i in range(0,len(rects)):
        x, y, w, h = rects[i]
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi, (24,24))
        vec[i] = roi
    return hogdesc(vec)


if __name__ == '__main__':

    arguments = docopt.docopt(__doc__)

    words = []
    for w in arguments['<string>'].split(' '):
        sc = SearchCriteria.parse(w)
        templates2d = templates.get_templates(sc.tokens)
        words.append({'word': w, 'sc': sc, 'templates2d': templates2d, 'matches': []})

    classifier = load_classifier()

    for i in arguments['<image>']:
        img = cv2.imread(i)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # sharpen image
        blur = cv2.GaussianBlur(img_grey, (-1,-1), 3)
        img_grey = cv2.addWeighted(img_grey, 1.5, blur, -0.5, 0)

        # text region features
        mser = cv2.MSER_create()
        points = mser.detectRegions(img_grey, None)
        char_rects = map(cv2.boundingRect, points)

        # filter aspect ratio
        char_rects = [r for r in char_rects if 0.5 < float(r[2])/float(r[3]) < 2]

        # label features
        start = time.time()
        print "Running classifier..."
        vec = prepare_detect(img_grey, char_rects)
        char_results = classifier.predict(vec)
        char_results = char_results.reshape(-1)
        print "took %ds" % (time.time() - start,)

        for w in words:
            sc = w['sc']
            templates2d = w['templates2d']
            matches = w['matches']

            # word features
            char_matches = np.in1d(char_results, list(sc.indexset()))
            matching_char_rects = np.array(char_rects, int)[char_matches]
            matching_char_results = char_results[char_matches]
            word_rects = []
            for result, rect in zip(matching_char_results, matching_char_rects):
                key = searchcriteria.get_label_value(int(result)).lower()
                for t in templates2d[key]:
                    word_rects.append(t.match2d(rect))

            # filter within image
            word_rects = [r for r in word_rects if not (r[0] < 0 or r[1] < 0 or r[2] > img.shape[1] or r[3] > img.shape[0])]

            start = time.time()
            print "Running OCR..."
            for r in word_rects:
                roi = img_grey[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
                text = ocr.ocr(roi)
                for m in templates.match(sc.tokens, text.lower()):
                    matches.append(r)
            print "took %ds" % (time.time() - start,)

        start = time.time()
        print "Matching words and lines..."
        line_matches = [[r] for r in words[0]['matches']]
        for idx in range(1, len(words)):
            tmp = []
            second_word_matches = words[idx]['matches']
            for lm in line_matches:
                for r2 in second_word_matches:
                    r1 = lm[-1]
                    if same_height(r1, r2) and (next_on_same_line(r1, r2) or on_consecutive_line(r1, r2)):
                        tmp.append(list(lm) + [r2])
            line_matches = tmp
        print "took %ds" % (time.time() - start,)

        for lm in line_matches:
            color = (randint(0,255), randint(0,255), randint(0,255))
            x, y, w, h = cv2.boundingRect(np.concatenate([utils.points(r) for r in lm]))
            img = cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

        cv2.namedWindow('Matches ' + i, cv2.WINDOW_NORMAL)
        cv2.imshow('Matches ' + i, img)

    cv2.waitKey()