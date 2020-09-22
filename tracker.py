# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:tracker.py
# software: PyCharm

import numpy as np
from utils import get_iou


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
        self.measurement = None

    def predict(self):
        self.age += 1

    def matching(self):
        self.age = 0
        self.tentative = False

    def label(self, index):
        self.index = index

    def update(self, mean, cov, measurement):
        self.mean = mean
        self.cov = cov
        self.measurement = measurement


def matching_cascade(tracks,
                     detections,
                     kalman_filter,
                     age=30,
                     init_age=3,
                     gating_threshold=9.4877,
                     iou_threshold=0.3):
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
        tracks:             a list
        detections:         a list
        kalman_filter:
        age:
        init_age:
        gating_threshold:
        iou_threshold:

    Returns:
        new_tracks

    """
    num_trackers = len(tracks)

    delete_index = []

    # starting tracking
    for i in range(num_trackers):
        tracker = tracks[i]

        # the last frame optimal estimation
        mean = tracker.mean
        cov = tracker.cov
        # predict the current estimation by transformation matrix
        mean_pred, cov_pred = kalman_filter.predict(mean, cov)
        tracker.update(mean_pred, cov_pred, None)

        # age = age + 1
        tracker.predict()

        if tracker.tentative and tracker.age <= init_age:
            maha_distances = kalman_filter.maha_distance(mean_pred, cov_pred, detections)
            min_distance = np.min(maha_distances)
            min_arg = np.argmin(maha_distances)
            if min_distance <= gating_threshold:
                # 1.set tracker.tentative = False and age = 0
                # 2.update distribution and measurement
                # 3.delete this detection in detections
                tracker.matching()
                # update prediction results by kalman filter
                new_mean, new_cov = kalman_filter.update(mean_pred, cov_pred, detections[min_arg])
                tracker.update(new_mean, new_cov, detections[min_arg])
                detections.pop(min_arg)

        if (not tracker.tentative) and tracker.age <= age:
            maha_distances = kalman_filter.maha_distance(mean_pred, cov_pred, detections)
            min_distance = np.min(maha_distances)
            min_arg = np.argmin(maha_distances)
            if min_distance <= gating_threshold:
                # 1.set tracker.tentative = False and age = 0
                # 2.update distribution and measurement
                # 3.delete this detection in detections
                tracker.matching()
                # update prediction results by kalman filter
                new_mean, new_cov = kalman_filter.update(mean_pred, cov_pred, detections[min_arg])
                tracker.update(new_mean, new_cov, detections[min_arg])
                detections.pop(min_arg)

        if tracker.tentative and tracker.age > init_age:
            delete_index.append(i)

        if (not tracker.tentative) and tracker.age > age:
            delete_index.append(i)

    # delete trackers

    new_tracks = []

    delete_set = set(delete_index)
    total_set = set(np.arange(num_trackers))
    remain_set = total_set - delete_set
    for k in remain_set:
        new_tracks.append(tracks[k])

    # IOU association on the set of unconfirmed and unmatched tracks of age n = 1
    for j, tracker in enumerate(new_tracks):
        tentative = tracker.tentative
        age = tracker.age
        mean_ = tracker.mean
        cov_ = tracker.cov

        if tentative and age == 1:
            if len(detections) > 0:
                tracker_measure = tracker.measurement
                ious = get_iou(tracker_measure, detections)
                max_iou = np.max(ious)
                max_arg = np.argmax(ious)
                if max_iou >= iou_threshold:
                    tracker.matching()
                    # update prediction results by kalman filter
                    new_mean, new_cov = kalman_filter.update(mean_, cov_, detections[max_arg])
                    tracker.update(new_mean, new_cov, detections[max_arg])
                    detections.pop(max_arg)

    # initialize unmatched detections
    if len(detections) > 0:
        for detection in detections:
            mean_init, cov_init = kalman_filter.initiate(detection)
            new_tracker = Tracker()
            new_tracker.mean = mean_init
            new_tracker.cov = cov_init
            new_tracker.measurement = detection
            new_tracks.append(new_tracker)

    return new_tracks
