# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:visualize.py
# software: PyCharm

import cv2
from utils import xyah2box
import numpy as np


def visualize_results(tracking_list,
                      height,
                      img):
    """visualize the tracking results.
       we only visualize the targets that have been confirmed.
    """

    font_scale = cv2.getFontScaleFromHeight(fontFace=6,
                                            pixelHeight=height) / 20

    num_tracker = len(tracking_list)

    if num_tracker > 0:
        for tracker in tracking_list:
            tentative = tracker.tentative
            measurement = xyah2box(tracker.measurement)
            measurement = measurement.astype('int')
            label = tracker.index

            if not tentative:
                # visualize bounding box and label
                retval, _ = cv2.getTextSize(str(label), 6, font_scale, 2)
                origin1 = measurement[0:2]
                origin2 = origin1 + retval
                text_origin = origin1 + np.array([0, retval[1]])
                cv2.rectangle(img, origin1, origin2, (255, 192, 203), thickness=-1)
                cv2.putText(img, str(label), text_origin, 6, font_scale, (255, 255, 255), 1)

                cv2.rectangle(img, measurement[0:2], measurement[2:4], (0, 0, 255), thickness=2)

    return img
