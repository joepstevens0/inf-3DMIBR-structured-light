from cmath import pi
import math
from typing import List

import utils
import numpy as np
import cv2

def load_views(view_paths: List[str]) -> List[np.array]:
    return [utils.load_images(view) for view in view_paths]

def show_captures(left_view: List[np.array], right_view: List[np.array]):
    window_size = (int(left_view[0].shape[1]/2), int(left_view[0].shape[0]/2))
    left_window_name = "Left view"
    right_window_name = "Right view"

    cv2.namedWindow(left_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(left_window_name, width=window_size[0], height=window_size[1])
    cv2.namedWindow(right_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(right_window_name, width=window_size[0], height=window_size[1])

    for image_left,  image_right in zip(left_view, right_view):
        cv2.imshow(left_window_name, image_left)
        cv2.imshow(right_window_name, image_right)

        key = cv2.waitKey(1000)
        if key == 27:
            break

    cv2.destroyAllWindows()

def gen_sin_pattern(img: np.array):
    x_len = img.shape[1]
    y_len = img.shape[0]

    # low frequency horizontal
    img_0 = np.zeros((y_len, x_len,3))
    img_1 = np.zeros((y_len, x_len,3))
    img_2 = np.zeros((y_len, x_len,3))

    # high freq horizontal
    img_3 = np.zeros((y_len, x_len,3))
    img_4 = np.zeros((y_len, x_len,3))
    img_5 = np.zeros((y_len, x_len,3))

    # low frequency vertical
    img_6 = np.zeros((y_len, x_len,3))
    img_7 = np.zeros((y_len, x_len,3))
    img_8 = np.zeros((y_len, x_len,3))

    # high freq vertical
    img_9 = np.zeros((y_len, x_len,3))
    img_10 = np.zeros((y_len, x_len,3))
    img_11 = np.zeros((y_len, x_len,3))

    for x in range(0,x_len):
        phase = (x/x_len)*pi*2
        img_0[:,x] = img[:,x]*np.full((y_len,1),math.cos(phase))
        img_1[:,x] = img[:,x]*np.full((y_len,1 ),math.cos(phase - np.deg2rad(120)))
        img_2[:,x] = img[:,x]*np.full((y_len,1),math.cos(phase + np.deg2rad(120)))
        img_3[:,x] = img[:,x]*np.full((y_len,1),math.cos(phase*16))
        img_4[:,x] = img[:,x]*np.full((y_len,1),math.cos(phase*16 - np.deg2rad(120)))
        img_5[:,x] = img[:,x]*np.full((y_len,1),math.cos(phase*16 + np.deg2rad(120)))
    for y in range(0,y_len):
        phase = (y/y_len)*pi*2
        img_6[y, :] = img[y,:]*np.full((x_len,1),math.cos(phase))
        img_7[y, :] = img[y,:]*np.full((x_len,1),math.cos(phase - np.deg2rad(120)))
        img_8[y, :] = img[y,:]*np.full((x_len,1),math.cos(phase + np.deg2rad(120)))
        img_9[y, :] = img[y,:]*np.full((x_len,1),math.cos(phase*16))
        img_10[y, :] = img[y,:]*np.full((x_len,1),math.cos(phase*16 - np.deg2rad(120)))
        img_11[y, :] = img[y,:]*np.full((x_len,1),math.cos(phase*16 + np.deg2rad(120)))
    
    return [img_0, img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8, img_9, img_10, img_11]

def show_sine_patterns():
    views = [
        "../dataset/Sinus/sinus_view0.xml",
        "../dataset/Sinus/sinus_view1.xml"
    ]
    left_view, right_view = load_views(views)

    sin_img_left = gen_sin_pattern(left_view[0].astype('float32')/255.)
    sin_img_right = gen_sin_pattern(right_view[0].astype('float32')/255.)

    show_captures(sin_img_left, sin_img_right)

if __name__ == "__main__":
    show_sine_patterns()