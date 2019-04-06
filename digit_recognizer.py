import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

from image_processing_tools import *
from digit_recognition_tools import *
from coord_operations import *


def recognize(img):
    im = np.copy(img)
    h, w, d = im.shape

    # resizeing
    dim = 75
    zoom_coef = min(dim/h, dim/w)
    im = zoom_bilinear_interpolation(to_gray(im), zoom_coef)
    # plt.imshow(im, cmap='gray')
    # plt.show()

    # negative
    im = negative(im)

    # thresholding (to binary)
    threshold = thresh_val(im)
    im = thresholding(im, threshold)
    plt.imshow(im, cmap='gray')
    plt.show()

    # closing, to remove irrelevant holes
    im = closing(im)
    # plt.imshow(im, cmap='gray')
    # plt.show()

    im = defect_filling(im)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.imsave('ss.png', 255*im, cmap='gray')

    # count the number of holes
    holes = get_holes(im)
    num_holes = len(holes) - 1
    print(' - number of holes:', num_holes)
    if num_holes == 2:
        return 8


    # majos/minor axis
    p1, p2 = major_axis(im)
    print(' - major axis points:', p1, p2)
    q1, q2 = minor_axis(im, (p1, p2))
    print(' - minor axis points:', q1, q2)
    if dist(q1, q2) < dim*0.1:
        return 1


    if num_holes == 1:
        h1, h2 = major_axis(holes[1])
        print(' - major axis of hole:', h1, h2)
        if dist(h1, h2) > dist(p1, p2)*0.75:
            return 0
        hm = mid(h1, h2)
        if dist(hm, p2) < dist(p1, p2)/2:
            if dist(h1, h2) > dist(q1, q2)/2:
                return 6
            else:
                return 2
        else:
            return 9