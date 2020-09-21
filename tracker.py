# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:tracker.py
# software: PyCharm

import numpy as np


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,  # we use four dimension space, so the threshold is 9.4877
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class Tracker:

    def __init__(self):
        self.age = 0
        self.tentative = True
        self.index = None
        self.mean = None
        self.cov = None

    def not_matching(self):
        self.age += 1

    def matching(self):
        self.age = 0
        self.tentative = False

    def label(self, index):
        self.index = index

    def update(self, mean, cov):
        self.mean = mean
        self.cov = cov


def matching_cascade(tracks,
                     detections,
                     kalman_filter,
                     age=30,
                     init_age=3,
                     gating_threshold=9.4877):
    """matching cascade
    tracking list: [1, 2, ..., N]
    detection list: [1, 2, ..., M]
    1. matching by maha_distance.
    2. if it is tentative, the max age is 3.
       if age > 3, just delete this tracker.
    3. if it has been matched, the max age is 30.
       if it has been matched, the age is set to 0.
       and we need to update the location of bounding boxes that have been matched.
       if it has not been matched, the age is added 1.

    Args:
        tracks:
        detections:
        kalman_filter:
        age:
        init_age:
        gating_threshold:

    Returns:

    """
    num_trackers = len(tracks)

    # starting tracking
    for i in range(num_trackers):
        tracker = tracks[i]
        mean = tracker.mean
        cov = tracker.cov
        if tracker.tentative and tracker.age < 3:
            maha_distances = kalman_filter.maha_distance(mean, cov, detections)
            min_distance = np.min(maha_distances)
            if min_distance <= gating_threshold:



