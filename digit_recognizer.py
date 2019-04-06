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
    plt.imshow(im, cmap='gray')
    plt.show()

    # thresholding (to binary)
    im = thresholding(negative(im))
    plt.imshow(im, cmap='gray')
    plt.show()

    # closing, to remove irrelevant holes
    im = closing(im)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.imsave('ss.png', 255*im, cmap='gray')

    # count the number of holes
    num_holes = number_of_holes(im)
    print('----')
    print(num_holes)


    p1, p2 = major_axis(im)
    print(p1, p2)

    q1, q2 = minor_axis(im, (p1, p2))
    print(q1, q2)