# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:detector.py
# software: PyCharm

import tensorflow as tf


class Detector:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.generate_model()

    def generate_model(self):
        # create model
        model = tf.saved_model.load(self.model_path)
        model = model.signatures['serving_default']
        return model

    def detect(self, inputs, image_shape):
        # inference
        boxes, scores, classes = self.model(input_2=inputs,
                                            input_3=image_shape)
        return boxes, scores, classes
