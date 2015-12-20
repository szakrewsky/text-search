__author__ = 'sz372'

import cv2
import numpy as np
import utils

class Line(object):

    def __init__(self, rects, text=[]):
        self.counts = None
        self.components = rects
        x, y, w, h = cv2.boundingRect(np.concatenate([utils.points(r) for r in rects]))
        self.rect = (x, y, w-1, h-1)
        self.wavg = sum((c[2] for c in self.components))/len(self.components)
        self.text = text

    def extends(self, line):
        x, y, w, h = self.rect
        lx, ly, lw, lh = line.rect
        if abs(h - lh) > min(h,lh):
            return False
        if abs(y - ly) > min(h,lh)/float(4):
            return False
        if abs(self.wavg - line.wavg) > self.wavg/float(2):
            return False
        return True

    def extend(self, line):
        self.components.extend(line.components)
        x, y, w, h = cv2.boundingRect(np.array(utils.points(self.rect) + utils.points(line.rect)))
        self.rect = (x, y, w-1, h-1)
        self.wavg = sum((c[2] for c in self.components))/len(self.components)
        self.text.extend(line.text)

    def filter(self):
        ordered = np.argsort(self.components, axis=0)[:,0]
        self.components = np.array(self.components)[ordered].tolist()
        self.text = np.array(self.text)[ordered].tolist()

        tmp_counts = []
        tmp = []
        t = None
        tmp_text = []

        for c,l in zip(self.components,self.text):
            if t is not None and abs(t[0] - c[0]) < 10 and abs(t[1] - c[1]) < 10 and abs(t[2] - c[2]) < 10 and abs(t[3] - c[3]) < 15:
                tmp_counts[-1] += 1
                continue
            elif t is not None and abs(t[0] - c[0]) < 10 and abs(t[1] - c[1]) < 10 and abs(t[3] - c[3]) < 15:
                if t[2] < c[2]:
                    tmp_counts[-1] += 1
                    continue
                else:
                    tmp[-1] = c
                    tmp_text[-1] = l
                    tmp_counts[-1] += 1
                    continue
            t = c
            tmp.append(t)
            tmp_text.append(l)
            tmp_counts.append(0)

        self.components = tmp
        self.text = tmp_text
        self.counts = tmp_counts


    def filter2(self):
        ordered = np.argsort(self.components, axis=0)[:,0]
        self.components = np.array(self.components)[ordered].tolist()
        self.text = np.array(self.text)[ordered].tolist()

        tmp_counts = []
        tmp = []
        t = None
        tmp_text = []

        for c,l in zip(self.components,self.text):
            if t is not None and abs(t[0] - c[0]) < 10 and abs(t[1] - c[1]) < 10 and abs(t[2] - c[2]) < 10 and abs(t[3] - c[3]) < 15:
                tmp_counts[-1] += 1
                continue
            elif t is not None and abs(t[0] - c[0]) < 10 and abs(t[1] - c[1]) < 10 and abs(t[3] - c[3]) < 15:
                if t[2] < c[2]:
                    tmp_counts[-1] += 1
                    continue
                else:
                    tmp[-1] = c
                    tmp_text[-1] = l
                    tmp_counts[-1] += 1
                    continue
            t = c
            tmp.append(t)
            tmp_text.append(l)
            tmp_counts.append(0)

        self.components = tmp
        self.text = tmp_text
        self.counts = tmp_counts



    def draw(self, img):
        img = img.copy()
        x, y, w, h = self.rect
        for c in self.components:
            cx, cy, cw, ch = c[0], c[1], c[2], c[3]
            img = cv2.rectangle(img,(cx,cy),(cx+cw,cy+ch),(255,0,0),1)
        cv2.namedWindow('Line', cv2.WINDOW_NORMAL)
        cv2.imshow('Line', img[y:y+h,x:x+w])
        cv2.waitKey()


def find_lines(rects, labels=[]):
    rects = [Line([r],[l]) for r,l in zip(rects,labels)]

    def _join_lines(rects):
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
                if a.extends(b):
                    overlap = True
                    a.extend(b)
                    used.append(j)

            results.append(a)
        return overlap, results

    overlap, results = _join_lines(rects)
    while overlap:
        overlap, results = _join_lines(results)
    return results
