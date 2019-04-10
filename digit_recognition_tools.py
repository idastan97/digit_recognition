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
    return list(filter(lambda x: x>0, np.sum(img, axis=1).tolist()))


def diff(arr):
    res = []
    for i in range(len(arr)-1):
        res.append(arr[i+1]-arr[i])
    return res


def draw_line_h(img, p1, p2, t=3):
    im = np.copy(img)
    h, w = im.shape
    a, b = line_equation(p1[::-1], p2[::-1])
    # # print(a, b)
    for i in range(1, w-1):
        j = ceil(a*i + b)
        for k in range(t):
            if j-k > 0 and j-k < h-1:
                im[j-k][i] = 1
    return im


def draw_line_v(img, p1, p2, t=3):
    im = np.copy(img)
    h, w = im.shape
    a, b = line_equation(p1, p2)
    # # print(a, b)
    for i in range(1, h-1):
        j = ceil(a*i + b)
        for k in range(t):
            if j+k > 0 and j+k < w-1:
                im[i][j+k] = 1
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
    h, w = img.shape
    im = np.ones((h, w), dtype=int)
    for i in range(h):
        for j in range(w-1, -1, -1):
            if img[i][j]:
                im[i][j] = 1
                break
            im[i][j] = 0
    return im


def left_edges(img):
    h, w = img.shape
    im = np.ones((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if img[i][j]:
                im[i][j] = 1
                break
            im[i][j] = 0
    return im


def right_maxs(img):
    im = right_edges(img)
    h, w = img.shape

    # lowest_p = lowest_point(im)
    # highest_p = highest_point(im)
    ru = erosion(im, np.array([[np.nan, 0, 0], [np.nan, 1, 0], [np.nan, np.nan, 0]]))
    rl = erosion(im, np.array([[np.nan, np.nan, 0], [np.nan, 1, 0], [np.nan, 0, 0]]))
    res = []
    
    av = [1]*h

    for j in range(w-1, -1, -1):
        previ = h
        for i in range(h-1, -1, -1):
            if ru[i][j]:
                for x in range(i, previ):
                    if not im[x][j] or not av[x]:
                        break
                    if rl[x][j]:
                        rl[x][j]=0
                        res.append((i, x, j))
                        av[i: x+1] = [0]*(x-i+1)
                        break
                previ = i
    return res


def left_maxs(img):
    # # print('*************')
    im = left_edges(img)
    h, w = img.shape

    # lowest_p = lowest_point(im)
    # highest_p = highest_point(im)

    ll = erosion(im, np.array([[0, np.nan, np.nan], [0, 1, np.nan], [0, 0, np.nan]]))
    lu = erosion(im, np.array([[0, 0, np.nan], [0, 1, np.nan], [0, np.nan, np.nan]]))
    res = []

    av = [1]*h
    
    for j in range(w):
        previ = h
        for i in range(h-1, -1, -1):
            if lu[i][j]:
                # # print(i, j)
                for x in range(i, previ):
                    if not im[x][j] or not av[x]:
                        break
                    if ll[x][j]:
                        ll[x][j]=0
                        res.append((i, x, j))
                        av[i: x+1] = [0]*(x-i+1)
                        break
                previ = i
    return res


def check_horizontal_line(img, p1, p2):
    a, b = line_equation(p1[::-1], p2[::-1])
    for j in range(p1[1]+1, p2[1]):
        i = round(a*j + b)
        adjs = [img[i-2][j], img[i-1][j], img[i][j], img[i+1][j], img[i+2][j]]
        if not any(adjs):
            # # print('---------')
            # # print(i, j)
            return False
    return True


def check_vertical_line(img, p1, p2):
    a, b = line_equation(p1, p2)
    for i in range(p1[0]+1, p2[0]):
        j = round(a*i + b)
        adjs = [img[i][j-2], img[i][j-1], img[i][j], img[i][j+1], img[i][j+2]]
        if not any(adjs):
            # # print('---------')
            # # print(i, j)
            return False
    return True


def horizontal_lines(img, thickness):
    rps = list(map(lambda p: ((p[0]+p[1])//2, p[2]), filter(lambda p: p[1] - p[0] <= thickness, right_maxs(img))))
    lps = list(map(lambda p: ((p[0]+p[1])//2, p[2]), filter(lambda p: p[1] - p[0] <= thickness, left_maxs(img))))
    res = []
    for rp in rps:
        l = None
        min_slope = inf
        for lp in lps:
            s = abs(slope(rp[::-1], lp[::-1]))
            if s <= 0.5 and s < inf and dist(rp, lp) > thickness and check_horizontal_line(img, lp, rp):
                l = lp
                min_slope = s
        if not (l is None):
            res.append((l, rp))
    return res


def widths(img):
    h, w = img.shape
    re = dists_from_right(img)
    le = dists_from_left(img)
    return [w - (re[i] + le[i])  for i in range(len(re))]

def lefter_point(img):
    h, w = img.shape
    for j in range(w):
        for i in range(h-1, -1, -1):
            if img[i][j]:
                return i, j
