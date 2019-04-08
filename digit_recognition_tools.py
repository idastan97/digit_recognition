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


def get_holes(img):
    ''' new implementation with BFS. Fast '''
    adjs = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1)
    ]
    holes = []
    im = np.copy(img)
    h, w = im.shape
    for i in range(h):
        for j in range(w):
            if im[i][j] == 0:
                hol = np.zeros((h, w), dtype=int)
                q = Queue()
                q.put((i, j))
                im[i][j] = 1
                hol[i][j]
                while not q.empty():
                    x, y = q.get()
                    for adj in adjs:
                        ax, ay = x+adj[0], y+adj[1]
                        if (ax >= 0 and ax < h and ay >= 0 and ay < w and
                            im[ax][ay] == 0):
                            q.put((ax, ay))
                            im[ax][ay] = 1
                            hol[ax][ay] = 1
                holes.append(hol)
    return holes


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
    res = (a, a)
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
                x, y = projection_point_to_segment((i, j), a, b)
                x = round(x)
                y = round(y)
                d = dist_sq((i, j), (x, y))
                if d > max_d:
                    max_d = d
                    res = ((i, j), (x, y))
    return res


def defect_filling(img):
    im = np.copy(img)
    h, w = im.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if im[i][j] == 0:
                a = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]
                for k in range(8):
                    if (im[a[k][0]][a[k][1]] and 
                        im[a[(k-2)%8][0]][a[(k-2)%8][1]]==0 and im[a[(k-1)%8][0]][a[(k-1)%8][1]]==0 and
                        im[a[(k+1)%8][0]][a[(k+1)%8][1]]==0 and im[a[(k+2)%8][0]][a[(k+2)%8][1]]==0 and
                        (im[a[(k+3)%8][0]][a[(k+3)%8][1]] or im[a[(k+4)%8][0]][a[(k+4)%8][1]] or im[a[(k+5)%8][0]][a[(k+5)%8][1]])):
                        im[i][j] = 1
                        break
    return im


def dists_from_left(img):
    h, w = img.shape
    res = []
    for i in range(h):
        for j in range(w):
            if img[i][j] == 1:
                res.append(j)
                break
    return res


def dists_from_right(img):
    h, w = img.shape
    res = []
    for i in range(h):
        for j in range(-1, -w-1, -1):
            if img[i][j] == 1:
                res.append(-j-1)
                break
    return res


def sum_horizontal(img):
    return np.sum(img, axis=1)


def diff(arr):
    res = []
    for i in range(len(arr)-1):
        res.append(arr[i+1]-arr[i])
    return res


def draw_line(img, p1, p2):
    im = np.copy(img)
    h, w = im.shape
    a, b = line_equation(p1[::-1], p2[::-1])
    print(a, b)
    for i in range(1, w-1):
        j = ceil(a*i + b)
        for k in range(3):
            if j-k >= 0 and j-k < h:
                im[j-k][i] = 1
    return im 


def lowest_point(img):
    h, w = img.shape
    for i in range(h-1, -1, -1):
        for j in range(w):
            if img[i][j]:
                return i, j


def highest_point(img):
    h, w = img.shape
    for i in range(h):
        for j in range(w-1, -1, -1):
            if img[i][j]:
                return i, j


def right_edges(img):
    im = np.copy(img)
    h, w = img.shape
    lowest_p = lowest_point(im)
    highest_p = highest_point(im)
    