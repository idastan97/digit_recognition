import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import plotly.plotly as py
import plotly.graph_objs as go

from image_processing_tools import *
from digit_recognition_tools import *
from coord_operations import *


def recognize(img):
    thickness = 5

    im = np.copy(img)
    h, w, d = im.shape

    # resizeing
    dim = 75
    zoom_coef = min(dim/h, dim/w)
    im = zoom_bilinear_interpolation(to_gray(im), zoom_coef)
    h, w = im.shape
    # plt.imshow(im, cmap='gray')
    # plt.show()

    # negative
    im = negative(im)

    # thresholding (to binary)
    threshold = thresh_val(im)
    im = thresholding(im, threshold)
    # plt.imshow(im, cmap='gray')
    # plt.show()

    # closing, to remove irrelevant holes
    im = closing(im)
    # plt.imshow(im, cmap='gray')
    # plt.show()

    im = defect_filling(im)
    plt.imshow(im, cmap='gray')
    plt.show()
    # plt.imsave('ss.png', 255*im, cmap='gray')

    # count the number of holes
    holes = get_holes(im)
    num_holes = len(holes) - 1
    print(' - number of holes:', num_holes)
    if num_holes == 2:
        return 8


    lowest_p = lowest_point(im)
    highest_p = highest_point(im)


    # majos/minor axis
    p1, p2 = major_axis(im)
    mj = dist(p1, p2)
    print(' - major axis ur:', p1, p2)
    q1, q2 = minor_axis(im, (p1, p2))
    print(' - minor axis ur:', q1, q2)
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

    ur = erosion(im, np.array([[0, 0, 0], [np.nan, 1, 0], [np.nan, np.nan, np.nan]]))
    # plt.imshow(ur, cmap='gray')
    # plt.show()
    vertices = []
    for i in range(h):
        for j in range(w):
            if ur[i][j]:
                if len(vertices) == 0:
                    vertices.append((i, j))
                elif abs(vertices[0][0] - i) < mj*0.3 and abs(vertices[0][1] - j) > 4:
                    vertices.append((i, j))
                    break
        if len(vertices) == 2:
            break
    if len(vertices) == 2:
        print(vertices[0], vertices[1])
        four = draw_line(im, vertices[0], vertices[1])
        plt.imshow(four, cmap='gray')
        plt.show()
        # count the number of holes
        fholes = get_holes(four)
        fnum_holes = len(fholes) - 1
        # print(' - number of holes:', num_holes)
        if fnum_holes == 1:
            return 4


    hls = horizontal_lines(im, thickness)
    print(hls)
    plt.imshow(im, cmap='gray')
    plt.show()


    # rl = points_to_list(erosion(im, np.array([[np.nan, np.nan, 0], [np.nan, 1, 0], [np.nan, 0, 0]])))[::-1]
    # # plt.imshow(rl, cmap='gray')
    # # plt.show()
    # ru = points_to_list(erosion(im, np.array([[np.nan, 0, 0], [np.nan, 1, 0], [np.nan, np.nan, 0]])))[::-1]
    # # plt.imshow(ru, cmap='gray')
    # # plt.show()
    # rp = None
    # print(lowest_p[0])
    # print(highest_p[0])
    # print(lowest_p[0]-(lowest_p[0]-highest_p[0])/3)
    # for el in rl:
    #     if lowest_p[0]-(lowest_p[0]-highest_p[0])/3 > el[0]:
    #         for eu in ru:
    #             if dist(el, eu) < thickness:
    #                 x, y = mid(el, eu)
    #                 rp = (round(x), round(y))
    #                 break
    #         if not (rp is None):
    #             break

    # ll = points_to_list(erosion(im, np.array([[0, np.nan, np.nan], [0, 1, np.nan], [0, 0, np.nan]])))[::-1]

    # # plt.imshow(ll, cmap='gray')
    # # plt.show()
    # lu = points_to_list(erosion(im, np.array([[0, 0, np.nan], [0, 1, np.nan], [0, np.nan, np.nan]])))[::-1]

    # # plt.imshow(lu, cmap='gray')
    # # plt.show()

    # lp = None
    # for el in ll:
    #     if lowest_p[0]-(lowest_p[0]-highest_p[0])/3 > el[0]:
    #         for eu in lu:
    #             if dist(el, eu) < thickness:
    #                 x, y = mid(el, eu)
    #                 lp = (round(x), round(y))
    #                 break
    #         if not (lp is None):
    #             break

    # print(rp, lp)

    # plt.imshow(im, cmap='gray')
    # plt.show()

    # ld = dists_from_left(im)
    # y_pos = np.arange(len(ld))
    # plt.bar(y_pos, ld, align='center', alpha=1.0)
    # plt.show()
    
    # rd = dists_from_right(im)
    # y_pos = np.arange(len(rd))
    # plt.bar(y_pos, rd, align='center', alpha=1.0)
    # plt.show()

    # sm = [w - ld[i] - rd[i] for i in range(len(ld))]
    # sm = diff(sum_horizontal(im))
    # y_pos = np.arange(len(sm))
    # plt.bar(y_pos, sm, align='center', alpha=1.0)
    # plt.show()

