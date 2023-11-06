from typing import List
from q4 import *
from q7 import *
from q5 import *
from pointcloud import *
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt


def id_hash(id: np.array) -> str:
    return str(id[0]) + str(id[1]) + str(id[2])

def load_matches() -> List[np.array]:
    print("Loading matches...")
    points1 = []
    points2 = []
    with open('../dataset/matches.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            row_a_splitted = row[0].split(',')
            row_b_splitted = row[1].split(',')
            points1.append([float(row_a_splitted[0]),float(row_a_splitted[1])])
            points2.append([float(row_b_splitted[0]),float(row_b_splitted[1])])
    return [np.array(points1), np.array(points2)]

def triangulate(projMatr1, projMatr2, projPoints1, projPoints2) -> np.array:
    points = []
    P1 = projMatr1
    P2 = projMatr2
    for i in range(0, projPoints1.shape[0]):
        point1 = projPoints1[i]
        point2 = projPoints2[i]

        A = [point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:]
        ]

        A = np.array(A, dtype=np.float32).reshape((4,4))
        A = A.transpose() @ A
        
        if i%10000 == 0:
            print(i)

        _,_,point_4d = np.linalg.svd(A, full_matrices = False)
        # print(point_4d[3,0:4])
        point_4d = point_4d[3,0:4]
        
        points.append(point_4d.flatten().tolist())

    return np.transpose(np.array(points))

depth_min = 0
depth_max = 0 

def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min



def calc_depth_extremes(cloud):
    global depth_min, depth_max
    depth_min = cloud[0][2]
    depth_max = cloud[0][2]
    for i in range(0, cloud.shape[0]):
        depth_min = min(depth_min, cloud[i][2])
        depth_max = max(depth_max, cloud[i][2])
    print(depth_min, depth_max)


def warp(image: np.array, points: np.array, points_4d: np.array,point_mask: np.array, cameraMatrix: np.array, dist) -> np.array:
    calc_depth_extremes(points_4d)

    depth = np.zeros(image.shape[0:2])
    mask = np.zeros(image.shape[0:2])
    for i in range(0, points.shape[0]):
        position = points[i]
        value = map(points_4d[i][2], depth_min, depth_max, 1, 0)
        depth[int(position[1]), int(position[0])] = value
        mask[int(position[1]), int(position[0])] = point_mask[i]

    rotation = np.append(np.identity(3), np.array([[0], [0], [0]]), axis=1).astype(np.float32)

    print(mask.shape, depth.shape, image.shape, rotation.shape, cameraMatrix.shape, dist.shape)
    
    warpedImage, warpedDepth, warpedMask = cv2.rgbd.warpFrame(image = image.astype(np.uint8), depth = depth, mask=mask, Rt = rotation, cameraMatrix = cameraMatrix, distCoeff=dist)
    
    return warpedImage

def calc_centrum(points: np.array) -> List[np.array]:
    min_x = np.min(points[:,0])
    min_y = np.min(points[:,1])
    min_z = np.min(points[:,2])

    max_x = np.max(points[:,0])
    max_y = np.max(points[:,1])
    max_z = np.max(points[:,2])

    return [np.array([min_x,min_y,min_z, 1]), np.array([max_x,max_y,max_z, 1])]

def show_3d(corners,points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(corners[:,0],corners[:,1],corners[:,2])
    ax.scatter(points[:,0], points[:,1],points[:,2])
    plt.show()

def project_virtual_corners(d_corners, projection_matrix, cameraMatrix:np.array, c_virtual_matrix: np.array) -> List[np.array]:
    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    cx = cameraMatrix[0,2]
    cy = cameraMatrix[1,2]

    inverse_virtual = np.linalg.inv(np.append(c_virtual_matrix,np.array([[0,0,0,1]]), axis=0))
    projection_mat4 = np.append(projection_matrix,np.array([[0,0,0,1]]), axis=0)
    
    plane_on_right = []
    for i in range(d_corners.shape[0]):
        point = inverse_virtual@projection_mat4@ np.append(d_corners[i], [1])
        corner_2d_x = ((point[0]*fx) /point[2]) + cx
        corner_2d_y = ((point[1]*fy) /point[2]) + cy
        plane_on_right.append([corner_2d_x,corner_2d_y])
    

    plane_on_left = []
    for i in range(d_corners.shape[0]):
        point = inverse_virtual @ np.append(d_corners[i], [1])
        corner_2d_x = ((point[0]*fx) /point[2]) + cx
        corner_2d_y = ((point[1]*fy) /point[2]) + cy
        plane_on_left.append([corner_2d_x,corner_2d_y])

    return [np.array(plane_on_left), np.array(plane_on_right)]
    

def calc_depth_plane(c_virtual_matrix: np.array, depth: float, shape: np.array, cameraMatrix:np.array) -> np.array:           
    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    cx = cameraMatrix[0,2]
    cy = cameraMatrix[1,2]

    corners = np.array([[0,0,0,1], [shape[1], 0, 0, 1], [shape[1], shape[0], 0, 1], [0, shape[0], 0, 1]])
    p1 = (c_virtual_matrix @ corners[0])
    p1[0] = ((p1[0] - cx)*depth)/fx
    p1[1] = ((p1[1] - cy)*depth)/fy
    p1[2] = depth
    p2 = (c_virtual_matrix @ corners[1])
    p2[0] = ((p2[0] - cx)*depth)/fx
    p2[1] = ((p2[1] - cy)*depth)/fy
    p2[2] = depth
    p3 = (c_virtual_matrix @ corners[2])
    p3[0] = ((p3[0] - cx)*depth)/fx
    p3[1] = ((p3[1] - cy)*depth)/fy
    p3[2] = depth
    p4 = (c_virtual_matrix @ corners[3])
    p4[0] = ((p4[0] - cx)*depth)/fx
    p4[1] = ((p4[1] - cy)*depth)/fy
    p4[2] = depth
    return np.array([p1,p2,p3,p4])

def calc_planesweep(depth: float,img_left: np.array, img_right: np.array, cameraMatrix: np.array, projection_matrix: np.array, c_virtual_matrix: np.array) -> List[np.array]:
    shape = img_left.shape
    
    depth_plane = calc_depth_plane(c_virtual_matrix, depth, shape, cameraMatrix)
    plane_on_left, plane_on_right = project_virtual_corners(depth_plane, projection_matrix, cameraMatrix, c_virtual_matrix)

    
    homography_left, mask = cv2.findHomography(plane_on_left,np.array([[0,0], [shape[1], 0], [shape[1], shape[0]], [0, shape[0]]]),  cv2.RANSAC)
    homography_right, mask = cv2.findHomography(plane_on_right,np.array([[0,0], [shape[1], 0], [shape[1], shape[0]], [0, shape[0]]]),  cv2.RANSAC)


    left_image = cv2.warpPerspective(img_left, M=homography_left, dsize=(shape[1], shape[0]))
    right_image =  cv2.warpPerspective(img_right, M=homography_right, dsize=(shape[1], shape[0]))

    return [left_image, right_image]

def error_func(left_sweeps: List[np.array], right_sweeps: List[np.array], correct_pixels) -> np.array:
    shape = left_sweeps[0].shape
    error = np.zeros(shape[0:2])
    for i in range(len(left_sweeps)):
        error += cv2.absdiff(left_sweeps[i], correct_pixels)
        error += cv2.absdiff(right_sweeps[i], correct_pixels)

    return error

def calc_new_viewpoint(img_left: np.array, img_right: np.array, depth_bounds: List[np.array], cameraMatrix: np.array, projection_matrix: np.array, c_virtual_matrix: np.array)-> List[np.array]:
    shape = img_left.shape
    min_point = (c_virtual_matrix @ depth_bounds[0])
    max_point = (c_virtual_matrix @ depth_bounds[1])

    min_depth = min_point[2]
    max_depth = max_point[2]
    depth = min_depth
    left_sweeps = []
    right_sweeps = []
    sweep_colors = []
    while depth < max_depth:
        left_image, right_image = calc_planesweep(depth, img_left, img_right, cameraMatrix, projection_matrix, c_virtual_matrix)
        left_sweeps.append(left_image)
        right_sweeps.append(right_image)
        sweep_colors.append((left_image + right_image)/2.)
        depth += 0.01

    # show_captures(sweeps_colors, sweeps_colors)

    errors = np.ones((shape[0],shape[1]))*np.inf
    correct_pixels = np.zeros((shape[0],shape[1], shape[2]), dtype=np.float32)

    for i in range(len(left_sweeps)):
        error = cv2.absdiff(left_sweeps[i], right_sweeps[i])
        error = error[:,:,0] + error[:,:,1] + error[:,:,2]
        error = cv2.filter2D(src = error, ddepth=-1, kernel=np.ones((3,3)))
        error[np.logical_and(left_sweeps[i][:,:,0] <0.1, np.logical_and(left_sweeps[i][:,:,1] < 0.1, left_sweeps[i][:,:,2] < 0.1))] += np.inf
        error[np.logical_and(right_sweeps[i][:,:,0] <0.1, np.logical_and(right_sweeps[i][:,:,1] < 0.1, right_sweeps[i][:,:,2] < 0.1))] += np.inf
        smaller_errors = error < errors # calc positions of smaller errors

        correct_pixels[smaller_errors] = sweep_colors[i][smaller_errors]
        errors[smaller_errors] = error[smaller_errors]  # update smallest errors

    return errors, correct_pixels

from scipy.spatial.transform import Rotation as R
from pyrr import Quaternion, Matrix33, Matrix44, Vector3, Vector4

def show_id_matches():
    views = [
        "../dataset/Sinus/sinus_view0.xml",
        "../dataset/Sinus/sinus_view1.xml",
        "../dataset/Sinus/sinus_chess.xml"
    ]
    left_view, right_view, checkerboards = load_views(views)

    for checkerboard in checkerboards:
        calc_chessboard_corners(checkerboard)

    shape = left_view[0].shape
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (shape[0], shape[1]), None, None)

    for i in range(0, len(left_view)):
        left_view[i] = left_view[i].astype('float32')/255.
    for i in range(0, len(right_view)):
        right_view[i] = right_view[i].astype('float32')/255.

    

    points1, points2 = load_matches()
   
    essential_matrix, mask = cv2.findEssentialMat(points1, points2,cameraMatrix, cv2.RANSAC)
    ret, rotation, translation, mask = cv2.recoverPose(essential_matrix, points1, points2,cameraMatrix, )

    projection_matrix = np.append(rotation, translation, axis=1)
    null_matrix = np.append(np.identity(3), np.array([[0], [0], [0]]), axis=1)
    
    points_4d = cv2.triangulatePoints(projMatr1 = cameraMatrix @ null_matrix, projMatr2 = cameraMatrix @ projection_matrix, projPoints1 = np.transpose(points1), projPoints2= np.transpose(points2))
    # points_4d = triangulate(projMatr1 = cameraMatrix @ null_matrix, projMatr2 = cameraMatrix @ projection_matrix, projPoints1 = points1, projPoints2= points2)
    points_4d = np.transpose(points_4d)
    points_4d[:,0] = points_4d[:,0]/points_4d[:,3]
    points_4d[:,1] = points_4d[:,1]/points_4d[:,3]
    points_4d[:,2] = points_4d[:,2]/points_4d[:,3]
    points_4d[:,3] = points_4d[:,3]/points_4d[:,3]

    images = []
    errors = []
    t = translation

    start = Quaternion.from_matrix(Matrix33.identity())
    end = Quaternion.from_matrix(rotation)

    i = 0.1
    while i < 1:
        c_virtual_matrix = np.append(start.slerp(end,i).matrix33, t*i, axis=1)
 
        
        min_point, max_point = calc_centrum(points_4d)

        error, correct_pixels = calc_new_viewpoint(left_view[0], right_view[0], [min_point, max_point], cameraMatrix, projection_matrix,c_virtual_matrix)
        images.append(correct_pixels)
        errors.append(error)

        i += 0.2


    show_captures(images,errors)

    

if __name__ == "__main__":
    show_id_matches()