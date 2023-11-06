from typing import List
from q4 import *
from q7 import *
from q5 import *
from pointcloud import *
import csv
from cv2 import sqrt
import numpy as np
import cv2

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
        
        point_4d = point_4d[3,0:4]
        
        points.append(point_4d.flatten().tolist())

    return np.transpose(np.array(points))
            
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
    
    show_point_cloud(points_4d, np.identity(4))

if __name__ == "__main__":
    show_id_matches()