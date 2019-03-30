import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing_tools import *

def number_of_holes(img):
    res = 0
    im = np.copy(img)
    h, w = im.shape
    while 1:
        rx, ry = -1, -1
        for i in range(h):
            for j in range(w):
                if im[i][j] == 0:
                    rx, ry = i, j
                    break
            if rx >= 0:
                break
        if rx < 0:
            break
        im = region_filling(im, rx, ry)
        res += 1
    return res-1


def recognize(img):
    dim = 75
    im = np.copy(img)
    h, w, d = im.shape
    zoom_coef = min(dim/h, dim/w)
    im = bilinear_interpolation(to_gray(im), zoom_coef)


    plt.imshow(im, cmap='gray')
    plt.show()
    im = thresholding(negative(im))
    plt.imshow(im, cmap='gray')
    plt.show()
    im = closing(im)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.imsave('ss.png', 255*im, cmap='gray')
    print(number_of_holes(im))