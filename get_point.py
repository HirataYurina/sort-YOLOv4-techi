# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:get_point.py
# software: PyCharm

import cv2

img = cv2.imread('./images/1.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (720, 480))


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    """get point when you click

    Args:
        event: mouse event
        x:     mouse point
        y:     mouse point
        flags:
        param:

    """
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey()
