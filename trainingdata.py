__author__ = 'sz372'

import cv2
import numpy as np
import os
import progressbar


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