__author__ = 'sz372'

import cv2
from features import hogdesc
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import trainingdata


def load_classifier(f='data/classifier.pkl', test=True, save=True):
    if os.path.isfile(f):
        print 'loading %s' % (f,)
        classifier = joblib.load(f)
    else:
        raw_data, raw_labels = trainingdata.load_raw(n=62*100)
        raw_data = map(lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), raw_data)
        raw_data = map(lambda i: cv2.resize(i, (24, 24)), raw_data)

        i = int(len(raw_data)*0.75)
        train_data = hogdesc(raw_data[0:i])
        test_data = hogdesc(raw_data[i:])
        train_labels = raw_labels[0:i]
        test_labels = raw_labels[i:]

        print 'training %d samples' % (len(train_data),)
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(train_data, train_labels)
        if test:
            print 'testing with %d samples' % (len(test_data),)
            print 'testing score %f' % (classifier.score(test_data, test_labels),)
        if save:
            joblib.dump(classifier, 'data/classifier.pkl')
            print 'classifier saved as %s' % (f,)
    return classifier
