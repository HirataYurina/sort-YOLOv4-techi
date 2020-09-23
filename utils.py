# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:utils.py
# software: PyCharm

import numpy as np
from PIL import Image
from tracker import Tracker


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


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def create_tracker(mean,
                   cov,
                   detection,
                   label_index):
    """create new tracker"""

    new_tracker = Tracker()
    new_tracker.mean = mean
    new_tracker.cov = cov
    new_tracker.measurement = detection
    # set label of this new tracker
    new_tracker.label(label_index)
    return new_tracker


def box2xyah(detections):
    """convert [x_min, y_min, x_max, y_max] to [x_center, y_center, aspect_ratio, height]

    Args:
        detections: the shape is (n, 4)

    Returns:
        detections

    """
    detections = np.reshape(detections, newshape=(-1, 4))
    center = (detections[..., 0:2] + detections[..., 2:4]) / 2  # (n, 2)
    boxes_wh = detections[..., 2:4] - detections[..., 0:2]  # (n, 2)
    aspect_ratio = boxes_wh[1] / boxes_wh[0]  # (n,)
    height = detections[..., -1]

    aspect_ratio = np.reshape(aspect_ratio, newshape=(-1, 1))
    height = np.reshape(height, newshape=(-1, 1))

    new_detections = np.concatenate([center, aspect_ratio, height], axis=-1)
    return new_detections


def xyah2box(target):
    """convert [x_center, y_center, aspect_ratio, height] to [x_min, y_min, x_max, y_max]

    Args:
        target: np.array
                the shape is (4,)

    Returns:
        new_target: np.array

    """
    height = target[3]
    ratio = target[2]
    width = height / ratio
    center = target[0:2]
    wh = np.array([width, height])
    xy_min = center - wh / 2
    xy_max = center + wh / 2

    return np.concatenate([xy_min, xy_max])
