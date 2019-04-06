import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from math import floor, ceil, sqrt

from image_processing_tools import *
from coord_operations import *


# def number_of_holes1(img):
#     ''' old implementation with region filling. Very long'''
#     res = 0
#     im = np.copy(img)
#     h, w = im.shape
#     while 1:
#         rx, ry = -1, -1
#         for i in range(h):
#             for j in range(w):
#                 if im[i][j] == 0:
#                     rx, ry = i, j
#                     break
#             if rx >= 0:
#                 break
#         if rx < 0:
#             break
#         im = region_filling(im, rx, ry)
#         res += 1
#     return res-1


def number_of_holes(img):
    ''' new implementation with BFS. Fast '''
    adjs = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1)
    ]
    res = 0
    im = np.copy(img)
    h, w = im.shape
    for i in range(h):
        for j in range(w):
            if im[i][j] == 0:
                q = Queue()
                q.put((i, j))
                im[i][j] = 1
                while not q.empty():
                    x, y = q.get()
                    for adj in adjs:
                        ax, ay = x+adj[0], y+adj[1]
                        if (ax >= 0 and ax < h and ay >= 0 and ay < w and
                            im[ax][ay] == 0):
                            q.put((ax, ay))
                            im[ax][ay] = 1
                res += 1
    return res - 1


def major_axis(img):
    res = None
    max_d = -1
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] == 1:
                for x in range(h):
                    for y in range(w):
                        if img[x][y] == 1:
                            d = dist_sq((i, j), (x, y))
                            if d > max_d:
                                max_d = d
                                res = ((i, j), (x, y))
    return res


def minor_axis(img, major_points):
    a, b = major_points
    res = None
    max_d = -1
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] == 1:
                for x in range(h):
                    for y in range(w):
                        if img[x][y] == 1 and is_orthogonal(a, b, (i, j), (x, y)):
                            d = dist_sq((i, j), (x, y))
                            if d > max_d:
                                max_d = d
                                res = ((i, j), (x, y))
    return res