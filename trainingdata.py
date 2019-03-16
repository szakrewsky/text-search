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


def load_raw_non_char(n):
    import cPickle
    fo = open('../coupon-db/cifar-10-batches-py/data_batch_1', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    temp = dict['data'][0:n]
    tempr = temp[:,0:1024].reshape((n,32,32))
    tempg = temp[:,1024:2048].reshape((n,32,32))
    tempb = temp[:,2048:3072].reshape((n,32,32))
    temp = np.empty((n,32,32,3), dtype='uint8')
    temp[:,:,:,0] = tempb
    temp[:,:,:,1] = tempg
    temp[:,:,:,2] = tempr
    return temp
