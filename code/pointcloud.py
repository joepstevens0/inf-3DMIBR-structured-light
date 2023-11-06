

import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
cloud = np.zeros((0,4))
view_matrix = np.identity(4, dtype=np.float32)
depth_min = 0
depth_max = 0 
rect = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])


SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min



def calc_depth_extremes():
    global cloud, depth_min, depth_max
    depth_min = cloud[0][2]
    depth_max = cloud[0][2]
    for i in range(0, cloud.shape[0]):
        if (cloud[i][2] < 0 or cloud[i][2] > 4):
            continue
        depth_min = min(depth_min, cloud[i][2])
        depth_max = max(depth_max, cloud[i][2])
    print(depth_min, depth_max)

def calc_color(point):
    return map(point[2], depth_min, depth_max, 0, 1)


def showScreen():
    global cloud, view_matrix, rect
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)
    glDisable(GL_CULL_FACE)

    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(view_matrix)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glPointSize(2)
    gluPerspective(45, (SCREEN_WIDTH / SCREEN_HEIGHT), 0.1, 50.0)
    glBegin(GL_POINTS)
    for i in range(0, cloud.shape[0], 15):
        p = cloud[i]
        color = calc_color(p)
        glColor3f(color, color, color)
        glVertex3f(p[0], -p[1], -p[2])
    glEnd()

    glBegin(GL_LINES)
    glColor3f(1, 1, 1)

    glVertex3f(rect[0,0],-rect[0,1],-rect[0,2])
    glVertex3f(rect[1,0],-rect[1,1],-rect[1,2])

    glVertex3f(rect[1,0],-rect[1,1],-rect[1,2])
    glVertex3f(rect[2,0],-rect[2,1],-rect[2,2])
    
    glVertex3f(rect[2,0],-rect[2,1],-rect[2,2])
    glVertex3f(rect[3,0],-rect[3,1],-rect[3,2])

    glVertex3f(rect[3,0],-rect[3,1],-rect[3,2])
    glVertex3f(rect[0,0],-rect[0,1],-rect[0,2])
    glEnd()
    glutSwapBuffers()

def keyboard_input(key,a,b):
    global view_matrix
    v = 0.1
    if key == b'a':
        view_matrix[3,0] += v
    if key == b'd':
        view_matrix[3,0] -= v
    if key == b'w':
        view_matrix[3,2] += v
    if key == b's':
        view_matrix[3,2] -= v

def set_depth_rect(depth_rect: np.array):
    global rect
    rect = depth_rect

def show_point_cloud(points, view):
    global cloud, view_matrix
    cloud = points
    view_matrix = view
    calc_depth_extremes()
    glutInit() # Initialize a glut instance which will allow us to customize our window
    glutInitDisplayMode(GLUT_RGBA) # Set the display mode to be colored
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT)   # Set the width and height of your window
    glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
    wind = glutCreateWindow("OpenGL Coding Practice") # Give your window a title
    glutDisplayFunc(showScreen)  # Tell OpenGL to call the showScreen method continuously
    glutIdleFunc(showScreen)     # Draw any graphics or shapes in the showScreen function at all times
    glutKeyboardFunc(keyboard_input)
    glutMainLoop()  # Keeps the window created above displaying/running in a loop
