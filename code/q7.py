
from typing import List
from q4 import *
import numpy as np
import cv2
import csv
import csv

def id_hash(id: np.array) -> str:
    return str(int(id[0]*255)) +"_"+ str(int(id[1]*255)) +"_"+ str(int(id[2]*255))

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
    return [np.array(points1).astype(np.int32), np.array(points2).astype(np.int32)]

def calc_matches(id_img_left: np.array, id_img_right: np.array) -> List[np.array]:
    shape = id_img_left.shape
    print(shape)
    
    points1 = []
    points2 = []

    value_indexes = {}

    for y in range(shape[0]):
        for x in range(shape[1]):
            left_id = id_hash(id_img_left[y,x])
            if "0_0_0" == left_id:
                continue
            if not left_id in value_indexes:
                value_indexes[left_id] = [(y,x)]
            else:
                value_indexes[left_id].append((y,x))
    
    count =0
    for y in range(id_img_right.shape[0]):
        for x in range(id_img_right.shape[1]):
            right_id = id_hash(id_img_right[y,x])
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


def show_id_matches():
    views = [
        "../dataset/Sinus/sinus_view0.xml",
        "../dataset/Sinus/sinus_view1.xml"
    ]
    left_view, right_view = load_views(views)

    for i in range(0, len(left_view)):
        left_view[i] = left_view[i].astype('float32')/255.
    for i in range(0, len(right_view)):
        right_view[i] = right_view[i].astype('float32')/255.

    id_img_left = gen_identifiers(left_view)[0]
    id_img_right = gen_identifiers(right_view)[0]


    points1, points2 = calc_matches(id_img_left, id_img_right)
    # points1, points2 = load_matches()
    
    with open('good_matches.csv', 'w') as file:
        for i in range(len(points1)):
            file.write(str(points1[i][0]) + "," + str(points1[i][1]) + ";" + str(points2[i][0]) + "," + str(points2[i][1]) + "\n")

    print("Finished writing matches to file")

    img1 = id_img_left
    img2 = id_img_right

    for i in range(0,len(points1)):
        color = (1, 0,0)
        img1 = cv2.circle(img1, tuple(points1[i]), radius=5, color=color, thickness=-1)
        img2 = cv2.circle(img2, tuple(points2[i]), radius=5, color=color, thickness=-1)


    show_captures([img1], [img2])

if __name__ == "__main__":
    show_id_matches()