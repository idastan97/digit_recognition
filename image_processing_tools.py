import cv2
import numpy as np
import matplotlib.pyplot as plt


def thresholding(img, threshold=128):
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


def morph(img, mask, op):
    im = np.copy(img)
    h, w = im.shape
    t, l = mask.shape
    t//=2
    l//=2
    im = np.concatenate( (np.full([h, l], np.nan), im, np.full([h, l], np.nan)), axis=1)
    im = np.concatenate( (np.full([t, w+2*l], np.nan), im, np.full([t, w+2*l], np.nan)), axis=0)
    res = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            x = i+t
            y = j+l
            res[i][j] = op(im, mask, x, y)
    return res


def dilation(img, mask):
    return morph(img, mask, hit)


def erosion(img, mask):
    return morph(img, mask, fit)


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
    B = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
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
    return erosion(dilation(img, s), s)