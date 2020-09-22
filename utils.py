# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:utils.py
# software: PyCharm

import numpy as np


def get_iou(measure, detections):
    """compute iou
    Args:
        measure:     [x, y, a, h]
        detections:  [[x, y, a, h], ...]
                     shape of (n, 4)

    Returns:
        ious

    """
    measure[2] = measure[3] / measure[2]
    detections[..., 2] = detections[..., 3] / detections[..., 2]

    measure_min = measure[0:2] - measure[2:4] / 2
    measure_max = measure[0:2] + measure[2:4] / 2
    detections_min = detections[..., 0:2] - detections[..., 2:4] / 2
    detections_max = detections[..., 0:2] + detections[..., 2:4] / 2

    measure_area = measure[2] * measure[3]
    detections_area = detections[..., 2] * detections[..., 3]

    insert_min = np.maximum(measure_min, detections_min)
    insert_max = np.minimum(measure_max, detections_max)
    insert_wh = insert_max - insert_min
    insert_area = np.maximum(0, insert_wh[..., 0] * insert_wh[..., 1])

    ious = insert_area / (measure_area + detections_area - insert_area)

    return ious
