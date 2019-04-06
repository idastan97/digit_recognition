import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from math import floor, ceil, sqrt


def to_gray(img):
    h, w, d = img.shape
    res = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            res[i][j] = round(np.sum(img[i][j])/3)
    return res


def negative(img):
    h, w = img.shape
    res = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            res[i][j] = 255 - img[i][j]
    return res


def thresholding(img, threshold=95):
    res = np.copy(img)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if res[i][j] < threshold:
                res[i][j] = 0
            else:
                res[i][j] = 1
    return res


def hit(img, mask, x, y):
    t, l = mask.shape
    t//=2
    l//=2
    for i in range(-t, t+1):
        mi = i+t
        for j in range(-l, l+1):
            mj = j+l
            if (not np.isnan(mask[mi][mj])) and mask[mi][mj] == img[x+i][y+j]:
                return 1
    return 0


def fit(img, mask, x, y):
    t, l = mask.shape
    t//=2
    l//=2
    for i in range(-t, t+1):
        mi = i+t
        for j in range(-l, l+1):
            mj = j+l
            if not np.isnan(mask[mi][mj]) and mask[mi][mj] != img[x+i][y+j]:
                return 0
    return 1


def spatial_filtering(img, mask, op, dtype=int):
    im = np.copy(img)
    h, w = im.shape
    t, l = mask.shape
    t//=2
    l//=2
    im = np.concatenate( (np.full([h, l], np.nan), im, np.full([h, l], np.nan)), axis=1)
    im = np.concatenate( (np.full([t, w+2*l], np.nan), im, np.full([t, w+2*l], np.nan)), axis=0)
    res = np.zeros((h, w), dtype=dtype)
    for i in range(h):
        for j in range(w):
            x = i+t

            y = j+l
            if dtype == int:
                res[i][j] = round(op(im, mask, x, y))
            else:
                res[i][j] = op(im, mask, x, y)
    return res


def dilation(img, mask=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])):
    return spatial_filtering(img, mask, hit)


def erosion(img, mask=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])):
    return spatial_filtering(img, mask, fit)


def intersection(A, B):
    h, w = A.shape
    res = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if A[i][j] == B[i][j] == 1:
                res[i][j] = 1
    return res


def union(A, B):
    h, w = A.shape
    res = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if A[i][j] == 1 or B[i][j] == 1:
                res[i][j] = 1
    return res


def complement(A):
    h, w = A.shape
    res = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if A[i][j] == 0:
                res[i][j] = 1
    return res


def region_filling(A, x, y):
    B = np.array([[np.nan, 1, np.nan], [1, 1, 1], [np.nan, 1, np.nan]])
    X = np.zeros(A.shape, dtype=int)
    X[x][y] = 1
    
    Ac = complement(A)
    X0 = None
    while not np.array_equal(X0, X):
        X0 = X
        X = intersection(dilation(X0, B), Ac)
    return union(X, A)


def closing(img):
    s = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return erosion(erosion(dilation(dilation(img))))


def linear_filter_at(img, mask, x, y):
    t, l = mask.shape
    t//=2
    l//=2
    res = 0.0
    for i in range(-t, t+1):
        mi = i+t
        for j in range(-l, l+1):
            mj = j+l
            if not np.isnan(mask[mi][mj]) and not np.isnan(img[x+i][y+j]):
                res += mask[mi][mj]*img[x+i][y+j]
    return res


def linear_filtering(img, mask, dtype=float):
    return spatial_filtering(img, mask, linear_filter_at, dtype=dtype)


def sobel(img, threshold=128):
    zI_horizontal = linear_filtering(img, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
    zI_vertical = linear_filtering(img,  np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
    h, w = img.shape
    res = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):

            p = math.sqrt( zI_vertical[i][j]**2 + zI_horizontal[i][j]**2 )
            if p > threshold:
                res[i][j] = p
    return res


def laplacian(img, threshold=128):
    mask = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

    res = linear_filtering(img, mask, dtype=int) 
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if res[i][j] < 0:
                res[i][j] = -res[i][j]
            if res[i][j] < threshold:
                res[i][j] = 0
    return res


def zoom_bilinear_interpolation(img, zoom_coef):
    h, w = img.shape
    zH = round(h*zoom_coef)
    zW = round(w*zoom_coef)
    zI = np.zeros((zH, zW), dtype=int)
    for i in range(zH):
        for j in range(zW):
            map_i = ((i*h)/zH + ((i-1)*h)/zH)/2
            map_j = ((j*w)/zW + ((j-1)*w)/zW)/2
            
            near_i1 = math.floor(map_i) + 0.5
            near_j1 = math.floor(map_j) + 0.5
            near_i2 = math.ceil(map_i) - 0.5
            near_j2 = math.ceil(map_j) - 0.5
            
            nears_i = [0, 0]
            nears_i_size = 0
            nears_j = [0, 0]
            nears_j_size = 0
            if map_i == floor(map_i) + 0.5:
                nears_i[0] = floor(map_i)+0.5
                nears_i_size = 1
            elif map_i < floor(map_i) + 0.5:
                if map_i >= 0:
                    nears_i[1] = floor(map_i)-0.5
                    nears_i_size = nears_i_size + 1
                nears_i[0] = floor(map_i)+0.5
                nears_i_size = nears_i_size + 1
            else:
                if map_i <= h-2:
                    nears_i[1] = ceil(map_i)+0.5
                    nears_i_size = nears_i_size + 1
                nears_i[0] = floor(map_i)+0.5
                nears_i_size = nears_i_size + 1
            
            if map_j == floor(map_j) + 0.5:
                nears_j[0] = floor(map_j)+0.5
                nears_j_size = 1;
            elif map_j < floor(map_j) + 0.5:
                if map_j >= 0:
                    nears_j[1] = floor(map_j)-0.5
                    nears_j_size = nears_j_size + 1
                nears_j[0] = floor(map_j)+0.5
                nears_j_size = nears_j_size + 1
            else:
                if map_j <= w-2:
                    nears_j[1] = ceil(map_j)+0.5
                    nears_j_size = nears_j_size + 1
                nears_j[0] = floor(map_j)+0.5
                nears_j_size = nears_j_size + 1
            
            dists = 0
            sm = 0
            for a in range(nears_i_size):
                for b in range(nears_j_size):
                    d = sqrt((nears_i[a]+map_i)**2 + (nears_j[b]+map_j)**2)
                    dists = dists + 1/d
                    sm = sm + (img[round(nears_i[a]+0.5)][round(nears_j[b]+0.5)]/d)
            zI[i][j] = round(sm/dists)
    return zI


def histogram(img):
    h, w = img.shape
    res = [0 for i in range(0, 256, 4)]
    for i in range(h):
        for j in range(w):
            res[img[i][j]//4] += 1
    return res


def thresh_val(img):
    h, w = img.shape
    hist = histogram(img)
    for i in range(len(hist)-1, -1,  -1):
        if hist[i] > 50:
            return (i+1)*4