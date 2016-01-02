__author__ = 'sz372'


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