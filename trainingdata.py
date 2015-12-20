__author__ = 'sz372'

import cv2
import numpy as np
import os
import progressbar

_NUMBERS =[chr(i) for i in range(48,58)]
_UPPER = [chr(i) for i in range(65,91)]
_LOWER = [chr(i) for i in range(97,123)]
_SAMPLES = _NUMBERS + _UPPER + _LOWER
LABELS = {_SAMPLES[i]: i + 1 for i in range(0, len(_SAMPLES))}
LABELS_INV = {i + 1: _SAMPLES[i] for i in range(0, len(_SAMPLES))}


def load_raw(n=0):
    rootdir = '../coupon-db/English/Fnt'
    files = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(rootdir) for file in files]
    if n > 0:
        files = np.random.choice(files, n)
    images = np.empty((len(files), 128, 128, 3), np.uint8)
    labels = np.empty((len(files)), int)

    print 'loading dataset of %d files' % (len(files),)
    bar = progressbar.ProgressBar(maxval=len(files), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i in range(0,len(files)):
        bar.update(i+1)
        images[i] = cv2.imread(files[i])
        labels[i] = int(files[i][-13:-10])

    bar.finish()
    print 'loaded %d samples' % (images.shape[0],)

    return images, labels