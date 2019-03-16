__author__ = 'sz372'

import cv2
from features import hogdesc, char_obj_features
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import svm
import trainingdata


n=100


def load_classifier(f='data/classifier.pkl', test=True, save=True):
    if os.path.isfile(f):
        print 'loading %s' % (f,)
        classifier = joblib.load(f)
    else:
        raw_data, raw_labels = trainingdata.load_raw(n=62*n)
        raw_data = map(lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), raw_data)
        raw_data = map(lambda i: cv2.resize(i, (24, 24)), raw_data)

        raw_non_char_data = trainingdata.load_raw_non_char(n=n)
        raw_non_char_data = map(lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), raw_non_char_data)
        raw_non_char_data = map(lambda i: cv2.resize(i, (24, 24)), raw_non_char_data)

        raw_data = np.concatenate([raw_data, raw_non_char_data])
        raw_labels = np.concatenate([raw_labels, [0]*n])

        mix = np.random.permutation(len(raw_labels))
        raw_data = raw_data[mix]
        raw_labels = raw_labels[mix]

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
            joblib.dump(classifier, f)
            print 'classifier saved as %s' % (f,)
    return classifier


def load_char_obj_classifier(f='char_obj/classifier.pkl', test=True, save=True):
    if os.path.isfile(f):
        print 'loading %s' % (f,)
        classifier = joblib.load(f)
    else:
        raw_data, raw_labels = trainingdata.load_raw(n=n)
        #raw_data = map(lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), raw_data)

        raw_non_char_data = trainingdata.load_raw_non_char(n=n)
        #raw_non_char_data = map(lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), raw_non_char_data)

        features = np.concatenate([char_obj_features(raw_data), char_obj_features(raw_non_char_data)])
        feature_labels = np.concatenate([[1]*n, [0]*n])

        mix = np.random.permutation(len(feature_labels))
        features = features[mix]
        feature_labels = feature_labels[mix]

        i = int(len(features)*0.75)
        train_data = features[0:i]
        test_data = features[i:]
        train_labels = feature_labels[0:i]
        test_labels = feature_labels[i:]

        print 'training %d samples' % (len(train_data),)
        classifier = svm.SVC()
        classifier.fit(train_data, train_labels)
        if test:
            print 'testing with %d samples' % (len(test_data),)
            print 'testing score %f' % (classifier.score(test_data, test_labels),)
        if save:
            joblib.dump(classifier, f)
            print 'classifier saved as %s' % (f,)
    return classifier