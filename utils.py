# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:utils.py
# software: PyCharm

import numpy as np
from PIL import Image
import tracker
import cv2
import copy


def get_iou(measure, detections):
    """compute iou
    Args:
        measure:     [x, y, a, h]
        detections:  [[x, y, a, h], ...]
                     shape of (n, 4)

    Returns:
        ious

    """
    detections = np.array(detections)
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
                   detection):
    """create new tracker"""

    new_tracker = tracker.MyTracker()
    new_tracker.mean = mean
    new_tracker.cov = cov
    new_tracker.measurement = detection
    return new_tracker


def box2xyah(detections):
    """convert [y_min, x_min, x_max, y_max] to [x_center, y_center, aspect_ratio, height]

    Args:
        detections: the shape is (n, 4)

    Returns:
        detections

    """
    # num = np.shape(detections)[0]
    detections = np.reshape(detections, newshape=(-1, 4))
    center = (detections[..., 0:2] + detections[..., 2:4]) / 2  # (n, 2)
    boxes_hw = detections[..., 2:4] - detections[..., 0:2]  # (n, 2)
    aspect_ratio = boxes_hw[..., 1] / boxes_hw[..., 0]  # (n,)
    # aspect_ratio = np.tile([0.618], reps=(num,))
    height = boxes_hw[..., 0]

    aspect_ratio = np.reshape(aspect_ratio, newshape=(-1, 1))
    height = np.reshape(height, newshape=(-1, 1))

    new_detections = np.concatenate([center[..., ::-1], aspect_ratio, height], axis=-1)
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
    width = height * ratio
    center = target[0:2]
    wh = np.array([width, height])
    xy_min = center - wh / 2
    xy_max = center + wh / 2

    return np.concatenate([xy_min, xy_max])


def transparent_region(img, point1, point2, point3, point4, color):
    polygon = np.array([point1, point2, point3, point4])

    img = cv2.resize(img, (720, 480))
    img_new = copy.deepcopy(img)
    # draw polygon
    cv2.fillConvexPoly(img_new, polygon, color)
    # transparent_img = alpha * img + (1 - alpha) * img_new
    img = cv2.addWeighted(img, 0.6, img_new, 0.4, 0.0)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return img


def judge_invasion(center,
                   point1,
                   point2,
                   point3,
                   point4):
    """judge invasion

    Args:
        center: center of bounding box
        point1: left top (x, y)
        point2: right top
        point3: right bottom
        point4: left bottom

    Returns:
        invasion: True or False

    """

    if point1[0] == point4[0] or point2[0] == point3[0]:
        raise ValueError('the edge should not be vertical')

    k_point12 = ((point2 - point1)[1]) / ((point2 - point1)[0])
    b_point12 = point1[1] - point1[0] * k_point12
    k_point43 = ((point4 - point3)[1]) / ((point4 - point3)[0])
    b_point43 = point4[1] - point4[0] * k_point43
    k_point23 = ((point2 - point3)[1]) / ((point2 - point3)[0])
    b_point23 = point2[1] - point2[0] * k_point23
    k_point14 = ((point4 - point1)[1]) / ((point4 - point1)[0])
    b_point14 = point1[1] - point1[0] * k_point14

    invasion = False
    if ((k_point12 * center[0] + b_point12 < center[1] and (k_point43 * center[0] + b_point43 > center[1]))
            and (((k_point23 * (k_point23 * center[0] + b_point23 - center[1])) < 0) and
                 ((k_point14 * (k_point14 * center[0] + b_point14 - center[1])) > 0))):
        invasion = True

    return invasion


if __name__ == '__main__':
    img_ = cv2.imread('./images/1.jpg', cv2.IMREAD_COLOR)
    transparent_region(img_,
                       (64, 235), (361, 166), (528, 256), (185, 361),
                       (0, 0, 255))
