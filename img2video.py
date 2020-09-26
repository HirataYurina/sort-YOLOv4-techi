# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:img2video.py
# software: PyCharm

import cv2
import os


def convert(img_path):
    """convert imgs to video"""
    img_names = [os.path.join(img_path, name) for name in os.listdir(img_path)]
    wirter_fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    writer = cv2.VideoWriter('person.avi', wirter_fourcc, 5, (720, 480))

    for img_name in img_names:
        frame = cv2.imread(img_name)
        frame = cv2.resize(frame, (720, 480))
        writer.write(frame)


if __name__ == '__main__':
    img_path_ = r'E:\Datasets\Crossing\img'
    convert(img_path_)
