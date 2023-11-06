from cmath import pi
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

        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()

def gen_phase_img(imgs: List[np.array]) -> np.array:
    I1 = (imgs[0][:,:,0] + imgs[0][:,:,1] + imgs[0][:,:,2])/3
    I2 = (imgs[1][:,:,0] + imgs[1][:,:,1] + imgs[1][:,:,2])/3
    I3 = (imgs[2][:,:,0] + imgs[2][:,:,1] + imgs[2][:,:,2])/3
    phase =  np.arctan2(np.sqrt(3)*(I1 - I3), (2*I2) - I1 - I3)
   
    return phase + pi

def phase_unwrap(high_freq_phase: np.array, low_freq_phase: np.array) -> np.array:
    repetition = np.floor((low_freq_phase/(2*pi))*16)
    return repetition*pi*2 + high_freq_phase

def gen_identifiers(imgs: List[np.array]):
    x_len = imgs[0].shape[1]
    y_len = imgs[0].shape[0]

    id_img = np.zeros((y_len, x_len,3))

    imgs[0] = (imgs[0][:,:,0] + imgs[0][:,:,1] + imgs[0][:,:,2])/3
    imgs[1] = (imgs[1][:,:,0] + imgs[1][:,:,1] + imgs[1][:,:,2])/3
    mask = (imgs[0]-imgs[1]) > 0.2

    horiz_high_freq = gen_phase_img([imgs[3], imgs[2], imgs[4]])
    horiz_low_freq = gen_phase_img([imgs[6], imgs[5], imgs[7]])

    vert_high_freq = gen_phase_img([imgs[9], imgs[8], imgs[10]])
    vert_low_freq = gen_phase_img([imgs[12], imgs[11], imgs[13]])

    id_img[:,:,1] = phase_unwrap(horiz_high_freq, horiz_low_freq)*mask

    id_img[:,:,2] = phase_unwrap(vert_high_freq, vert_low_freq)*mask
    
    id_img = (id_img)/(2*pi*16)


    print(np.min(id_img), np.max(id_img))
    
    return [id_img]

def show_sine_patterns():
    views = [
        "../dataset/Sinus/sinus_view0.xml",
        "../dataset/Sinus/sinus_view1.xml"
    ]
    left_view, right_view = load_views(views)

    for i in range(0, len(left_view)):
        left_view[i] = left_view[i].astype('float32')/255.
    for i in range(0, len(right_view)):
        right_view[i] = right_view[i].astype('float32')/255.

    sin_img_left = gen_identifiers(left_view)
    sin_img_right = gen_identifiers(right_view)

    show_captures(sin_img_left, sin_img_right)

if __name__ == "__main__":
    show_sine_patterns()