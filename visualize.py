# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:visualize.py
# software: PyCharm

import cv2
from utils import xyah2box, transparent_region, judge_invasion
import numpy as np
import copy


POINT1 = np.array((64, 235))
POINT2 = np.array((361, 166))
POINT3 = np.array((528, 256))
POINT4 = np.array((185, 361))


def visualize_results(tracking_list,
                      height,
                      img):
    """visualize the tracking results.
       we only visualize the targets that have been confirmed.
    """
    img_copy = copy.deepcopy(img)
    img = transparent_region(img,
                             POINT1, POINT2, POINT3, POINT4,
                             (0, 0, 255))
    font_scale = cv2.getFontScaleFromHeight(fontFace=6,
                                            pixelHeight=int(height)) / 25

    num_tracker = len(tracking_list)

    if num_tracker > 0:
        for tracker in tracking_list:
            tentative = tracker.tentative

            # if target is in the danger region, system givea an alarm.
            center = tracker.measurement[:2]
            invasion = judge_invasion(center, POINT1, POINT2, POINT3, POINT4)
            print(invasion)
            if invasion:
                img = transparent_region(img_copy,
                                         POINT1, POINT2, POINT3, POINT4,
                                         (255, 0, 0))

            measurement = xyah2box(tracker.measurement)
            measurement = measurement.astype('int')
            label = tracker.index

            if not tentative:
                # print(label)
                # visualize bounding box and label
                retval, _ = cv2.getTextSize(str(label), 6, font_scale, 2)
                origin1 = measurement[0:2]
                origin2 = origin1 + retval
                text_origin = origin1 + np.array([0, retval[1]])
                cv2.rectangle(img, (origin1[0], origin1[1]),
                              (origin2[0], origin2[1]), (255, 192, 203),
                              thickness=-1)
                cv2.putText(img, str(label), (text_origin[0], text_origin[1]),
                            6, font_scale, (255, 255, 255), 1)

                cv2.rectangle(img, (measurement[0], measurement[1]),
                              (measurement[2], measurement[3]), (0, 0, 255),
                              thickness=2)

    return img
