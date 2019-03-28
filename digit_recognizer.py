import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing_tools import thresholding, dilation, region_filling, closing

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
    im = np.copy(img)
    im = thresholding(im)
    plt.imshow(im)
    plt.show()

    print(number_of_holes(im))