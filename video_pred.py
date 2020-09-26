# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:video_pred.py
# software: PyCharm

from detector import Detector
from tracker import matching_cascade
import numpy as np
import cv2
from PIL import Image
from utils import letterbox_image, create_tracker, box2xyah
import tensorflow as tf
from kalman_filter import KalmanFilter
# from official_code.kalman_filter import KalmanFilter
from visualize import visualize_results

# the input size
INPUT_SIZE = [416, 416]


def main(video_path,
         model_path,
         track_target=0,
         visualize=True):
    """run video prediction

    Args:
        video_path:     video path
        model_path:     model path
        track_target:   0-person; 1-bicycle; 2-car; 7-truck
        visualize:      whether visualize tracking list

    """

    detector = Detector(model_path=model_path)
    kalman_filter = KalmanFilter()
    capture = cv2.VideoCapture(video_path)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    # tracking list
    tracking_list = []
    label_count = 0
    is_first_frame = True

    while True:
        success, frame = capture.read()

        if not success:
            capture.release()
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert to Image object
        frame_pil = Image.fromarray(np.uint8(frame))
        new_frame = letterbox_image(frame_pil, INPUT_SIZE)
        image_array = np.expand_dims(np.array(new_frame, dtype='float32') / 255.0, axis=0)
        image_shape = np.expand_dims(np.array([height, width], dtype='float32'), axis=0)
        image_constant = tf.constant(image_array, dtype=tf.float32)
        image_shape = tf.constant(image_shape, dtype=tf.float32)

        # detect image
        results = detector.detect(image_constant, image_shape)
        pred_results = []
        for key, value in results.items():
            pred_results.append(value)
        boxes = pred_results[0].numpy()
        # scores = scores.numpy
        classes = pred_results[2].numpy()

        # find tracking targets
        track_id = np.where(classes == track_target)[0]
        track_boxes = boxes[track_id]
        num_tracks = len(track_boxes)
        if num_tracks > 0:
            track_boxes = box2xyah(track_boxes)
            track_boxes = [track_box for track_box in track_boxes]

        if not is_first_frame:
            # start tracking
            tracking_list, label_count = matching_cascade(tracking_list, track_boxes,
                                                          kalman_filter, label_count)

        if is_first_frame and (num_tracks > 0):
            is_first_frame = False

            for i in range(num_tracks):
                # initialize first frame
                mean_init, cov_init = kalman_filter.initiate(measurement=track_boxes[i])
                # create tracker
                new_tracker = create_tracker(mean=mean_init,
                                             cov=cov_init,
                                             detection=track_boxes[i])
                tracking_list.append(new_tracker)

        if visualize:
            # visulize results
            img = visualize_results(tracking_list, height, frame)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('avoid invasion', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                capture.release()
                break


if __name__ == '__main__':

    # test my algorithm
    test_video_path = './video/person.avi'
    model_path = './saved_model_coco'
    main(test_video_path, model_path, visualize=True)
