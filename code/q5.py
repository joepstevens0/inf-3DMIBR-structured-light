from typing import List

import utils
import numpy as np
import cv2

checkerboard_dimensions = (9,7)

objp = np.zeros((checkerboard_dimensions[0]*checkerboard_dimensions[1],3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_dimensions[0],0:checkerboard_dimensions[1]].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

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

def calc_chessboard_corners(img: np.array):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_dimensions, None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # # Draw and display the corners
        # cv2.drawChessboardCorners(img, checkerboard_dimensions, corners, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

def undistort(img: np.array, mtx, dist) -> np.array:
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    undistorded = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    return undistorded

def calibrate():
    views = ["../dataset/Sinus/sinus_chess.xml"]
    checkerboards, = load_views(views)

    for checkerboard in checkerboards:
        calc_chessboard_corners(checkerboard)
    
    img = checkerboards[3]
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[0], img.shape[1]), None, None)

    undistorted = undistort(img, mtx,dist)

    show_captures([undistorted], [img])

if __name__ == "__main__":
    calibrate()