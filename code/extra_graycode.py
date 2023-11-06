from cmath import pi
from typing import List
import utils
import numpy as np
import cv2

def calc_matches(id_img_left: np.array, id_img_right: np.array) -> List[np.array]:
    shape = id_img_left.shape
    print(shape)
    
    points1 = []
    points2 = []

    value_indexes = {}

    for y in range(shape[0]):
        for x in range(shape[1]):
            left_id = id_img_left[y,x]
            if 0 == left_id:
                continue
            if not left_id in value_indexes:
                value_indexes[left_id] = [(y,x)]
            else:
                value_indexes[left_id].append((y,x))
    
    count =0
    for y in range(id_img_right.shape[0]):
        for x in range(id_img_right.shape[1]):
            right_id = id_img_right[y,x]
            if right_id in value_indexes:
                count +=1
                # index with same id
                p1_match = value_indexes[right_id]
                
                for j in range(0,1):
                    p = p1_match[j]
                    points2.append([x,y])
                    points1.append([p[1], p[0]])
    
    print("count" +str(count))
    return [np.array(points1).astype(np.int32), np.array(points2).astype(np.int32)]

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

def gen_bitmap(img: np.array, light_img: np.array) -> np.array:
    I = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
    return I < (light_img[:,:,0] + light_img[:,:,1] + light_img[:,:,2])/3 - 0.2

def gen_id_img(imgs: List[np.array]) -> np.array:
    shape = imgs[0].shape
    result = np.zeros(shape[0:2], dtype=np.uint64)
    for i in range(2, len(imgs)):
        bitmap = gen_bitmap(imgs[i], imgs[0])
        result += bitmap.astype(np.uint64)
        result = np.left_shift(result, np.ones(shape[0:2], dtype=np.uint64))

    print(np.unique(result).shape, result.shape[0]*result.shape[1])
    return result.astype(np.float64)/np.max(result)


def show_sine_patterns():
    views = [
        "../dataset/GrayCodes/graycodes_view0.xml",
        "../dataset/GrayCodes/graycodes_view1.xml"
    ]
    left_view, right_view = load_views(views)

    for i in range(0, len(left_view)):
        left_view[i] = left_view[i].astype('float32')/255.
    for i in range(0, len(right_view)):
        right_view[i] = right_view[i].astype('float32')/255.

    id_img_left = gen_id_img(left_view)
    id_img_right = gen_id_img(right_view)

    points1, points2 = calc_matches(id_img_left, id_img_right)

    with open('graycode_matches.csv', 'w') as file:
        for i in range(len(points1)):
            file.write(str(points1[i][0]) + "," + str(points1[i][1]) + ";" + str(points2[i][0]) + "," + str(points2[i][1]) + "\n")

    print("Finished writing matches to file")

    show_captures([id_img_left.astype(np.float64)], [id_img_right.astype(np.float64)])

if __name__ == "__main__":
    show_sine_patterns()