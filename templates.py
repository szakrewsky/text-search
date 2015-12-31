__author__ = 'sz372'

import nltk
import searchcriteria

class Match(object):

    def __init__(self, window):
        self.window = window

    def __cmp__(self, other):
        return cmp(self.window,other.window)

    def __repr__(self):
        return "%s(%s)" % (self.__class__, self.window)

    def __hash__(self):
        return hash(self.window)


class Template(object):

    def __init__(self, s, si):
        self.s = s
        self.si = si

    def match(self, labels, li):
        matches = []
        window = self._window(labels, li)
        c = nltk.edit_distance(self.s, window)
        c = c - (len(window) - len(self.s))
        if c < len(self.s)/float(2):
            matches.append(Match(window))
        return matches

    def _window(self, labels, li, e = 2):
        w_size = len(self.s) * e
        w_si = self.si * e

        start_offset = - w_si
        end_offset = w_size - w_si
        s, e = (max(0, li + start_offset), min(len(labels), li + end_offset))
        return labels[s:e]

    def match2d(self, rect):
        x,y,w,h = rect
        return x - (self.si * w) - 10, y - 10, len(self.s) * w + 20, h + 20


def match(s, labels):
    templates = {}
    for i,c in enumerate(s):
        if c not in templates:
            templates[c] = []
        templates[c].append(Template(s,i))

    matches = []
    for i,l in enumerate(labels):
        if l in templates:
            candidates = templates[l]
            for c in candidates:
                matches.extend(c.match(labels, i))
        if l in searchcriteria._NUMBERS and '\d' in templates:
            candidates = templates['\d']
            for c in candidates:
                matches.extend(c.match(labels, i))

    return set(matches)


def get_templates(s):
    templates = {}
    for i,c in enumerate(s):
        if c not in templates:
            templates[c] = []
        templates[c].append(Template(s,i))
    return templates


def template2d():
    for c, rect in zip(result, rects):
        key = trainingdata.LABELS_INV[int(c)].lower()
        rectarray = charmap.get(key, [])
        rectarray.append(rect)
        charmap[key] = rectarray

    for key in charmap:
        rectarray = charmap[key]
        line = textlines.Line(rectarray, [key]*len(rectarray))
        line.filter()
        charmap[key] = line.components

    templates2d = templates.get_templates(STRING.lower())
    for key in charmap:
        temparray = []
        rectarray = charmap.get(key)
        candidates = templates2d[key]
        for r in rectarray:
            for c in candidates:
                    temparray.append(c.match2d(r))
        charmap[key] = temparray

    temparray = list(itertools.chain(*charmap.itervalues()))
    line = textlines.Line(temparray, ['x']*len(temparray))
    line.filter2()

    return np.array(line.components)[np.array(line.counts) > 0]
