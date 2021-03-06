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
            r, g, b = img[i][j]
            res[i][j] = round(np.sum(img[i][j])/3)
            # res[i][j] = round(sqrt((r**2+g**2+b**2)/3))
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


def spatial_filtering(img, mask, op, dtype=int, mp=None):
    im = np.copy(img)
    h, w = im.shape
    t, l = mask.shape
    t//=2
    l//=2
    # im = np.concatenate( (np.full([h, l], np.nan), im, np.full([h, l], np.nan)), axis=1)
    # im = np.concatenate( (np.full([t, w+2*l], np.nan), im, np.full([t, w+2*l], np.nan)), axis=0)
    res = np.zeros((h, w), dtype=dtype)
    for i in range(t, h-t):
        for j in range(l, w-l):
            if mp is None or mp[i][j]:
                if dtype == int:
                    res[i][j] = round(op(im, mask, i, j))
                else:
                    res[i][j] = op(im, mask, i, j)
    return res


def dilation(img, mask=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), mp=None):
    return spatial_filtering(img, mask, hit, mp=mp)


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


def sobel(img, threshold=90):
    zI_horizontal = linear_filtering(img, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
    zI_vertical = linear_filtering(img,  np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
    h, w = img.shape
    res = np.zeros((h, w), dtype=int)
    for i in range(1, h-1):
        for j in range(1, w-1):

            p = math.sqrt( zI_vertical[i][j]**2 + zI_horizontal[i][j]**2 )
            if p > threshold:
                res[i][j] = 1
    return res


def laplacian(img, threshold=128):
    mask = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

    res = linear_filtering(img, mask, dtype=int) 
    h, w = img.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
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
            zI[i][j] = 1 if round(sm/dists) >= 1 else 0
    return zI


def histogram(img):
    h, w = img.shape
    res = [0 for i in range(0, 256, 4)]
    for i in range(h):
        for j in range(w):
            res[img[i][j]//4] += 1
    return res


def thresh_val(img):
    return (np.max(img) + np.min(img))*2.0/5.0


def points_to_list(img):
    h, w = img.shape
    res = []
    for i in range(h):
        for j in range(w):
            if img[i][j]:
                res.append((i, j))
    return res


def get_subarray(img, p1, p2):
    res = np.zeros(( p2[0]-p1[0]+1,  p2[1]-p1[1]+1), dtype=int)
    for i in range(p2[0]-p1[0]+1):
        for j in range(p2[1]-p1[1]+1):
            res[i][j] = img[i+p1[0]][j+p1[1]]
    return res


def add_frame(img, thick=3):
    h, w = img.shape
    im = img
    l = thick
    t = thick
    im = np.concatenate( (np.zeros([h, l], dtype=int), im, np.zeros([h, l], dtype=int)), axis=1)
    im = np.concatenate( (np.zeros([t, w+2*l], dtype=int), im, np.zeros([t, w+2*l], dtype=int)), axis=0)
    return im


def transpose(img):
    h, w = img.shape
    res = np.zeros((w, h), dtype=int)
    for i in range(h):
        for j in range(w):
            res[j][i] = img[i][j]
    return res

def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

