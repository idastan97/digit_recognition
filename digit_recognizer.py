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
    im = np.copy(img)
    h, w, d = im.shape

    
    
    im = to_gray(im)
    h, w = im.shape

    # negative
    im = negative(im)

    # thresholding (to binary)
    threshold = thresh_val(im)
    im = thresholding(im, threshold)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.imsave('ss.png', im, cmap='gray')

    adjs = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1)
    ]
    nums = []
    im2 = np.copy(im)
    # h, w = im.shape
    max_pxs = 0
    pxss = []
    cords = []
    for j in range(w):
        for i in range(h):
            if im[i][j] == 1:
                num = np.zeros((h, w), dtype=int)
                q = Queue()
                q.put((i, j))
                im[i][j] = 0
                num[i][j] = 1
                pxs = 1
                minx = i
                miny = j
                maxx = i
                maxy = j
                while not q.empty():
                    x, y = q.get()
                    for adj in adjs:
                        ax, ay = x+adj[0], y+adj[1]
                        if (ax >= 0 and ax < h and ay >= 0 and ay < w and
                            im[ax][ay] == 1):
                            q.put((ax, ay))
                            im[ax][ay] = 0
                            num[ax][ay] = 1
                            pxs += 1
                            if ax < minx:
                                minx = ax
                            if ax > maxx:
                                maxx= ax
                            if ay < miny:
                                miny = ay
                            if ay > maxy:
                                maxy= ay
                if pxs > max_pxs:
                    max_pxs = pxs
                pxss.append(pxs)
                cords.append(  ( (minx, miny), (maxx, maxy) )  )
                nums.append(num)
    nums2 = []
    for i in range(len(pxss)):
        if pxss[i] < max_pxs/4:
            continue
        p1, p2 = cords[i]
        # print(p1, p2)
        nums2.append(add_frame(get_subarray(im2, p1, p2)))
        # plt.imshow(nums2[-1], cmap='gray')
        # plt.show()

    res = []
    for i in range(len(nums2)):
        print(i)
        res.append(digit_recognize(nums2[i]))

    return res


def digit_recognize(img):
    thickness = 5

    h, w = img.shape
    # resizeing
    dim = 75
    zoom_coef = min(dim/h, dim/w)
    im = zoom_bilinear_interpolation(img, zoom_coef)

    h, w = im.shape

    # closing, to remove irrelevant holes
    im = closing(im)
    # # plt.imshow(im, cmap='gray')
    # # plt.show()

    im = defect_filling(im)
    plt.imshow(im, cmap='gray')
    plt.show()
    # plt.imsave('ss.png', im, cmap='gray')

    # count the number of holes
    holes = get_holes(im)
    num_holes = len(holes) - 1
    # # print(' - number of holes:', num_holes)
    if num_holes == 2:
        return 8


    lowest_p = lowest_point(im)
    highest_p = highest_point(im)


    # majos/minor axis
    p1, p2 = major_axis(im)
    mj = dist(p1, p2)
    # # print(' - major axis ur:', p1, p2)
    q1, q2 = minor_axis(im, (p1, p2))
    # # print(' - minor axis ur:', q1, q2)
    if dist(q1, q2) < dim*0.1:
        return 1


    if num_holes == 1:
        h1, h2 = major_axis(holes[1])
        # # print(' - major axis of hole:', h1, h2)
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
    # # plt.imshow(ur, cmap='gray')
    # # plt.show()
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
        # # print(vertices[0], vertices[1])
        four = draw_line_h(im, vertices[0], vertices[1])
        # plt.imshow(four, cmap='gray')
        # plt.show()
        # count the number of holes
        fholes = get_holes(four)
        fnum_holes = len(fholes) - 1
        # print(' - number of holes:', num_holes)
        if fnum_holes == 1:
            return 4


    hls = horizontal_lines(im, thickness)
    for lp, rp in hls:
        m = mid(lp, rp)
        d = lowest_p[0] - highest_p[0]
        if highest_p[0] + 0.3*d <= m[0] <= lowest_p[0] - 0.3*d:
            return 7
    # # print(hls)
    # plt.imshow(im, cmap='gray')
    # plt.show()


    lps = sorted(list(map(lambda p: ((p[0]+p[1])//2, p[2]), left_maxs(im))))
    # print(lps)
    d = lowest_p[0] - highest_p[0]
    if (len(lps) == 3 and 
        highest_p[0] + 0.25*d <= lps[1][0] <= lowest_p[0] - 0.25*d and 
        not check_vertical_line(im, lps[0], lps[1]) and 
        not check_vertical_line(im, lps[1], lps[2])):
        return 3

    vsum = list(filter(lambda a: a>0, widths(im) ))
    line_lo = 0
    for i in range(-1, -len(vsum)-1, -1):
        if vsum[i] <= thickness:
            line_lo += 1
        else:
            break
    # # print('[[[[[[[[[[')
    # # print(line_lo)
    # # print(d)

    if line_lo > d*0.7:
        return 7
    if line_lo > d*0.35:
        return 1


    # mid_p = mid(lowest_p, highest_p)
    rds = dists_from_right(im)[3:]
    lds = dists_from_left(im)[:-3]
    wmx = w - min(rds) - min(lds)
    
    mx1 = max(rds[:len(rds)//2])
    
    mx2 = max(lds[len(lds)//2:])
    # print('wmx', wmx)
    # print('mx1', mx1)
    # print('mx2', mx2)
    if (mx2+mx1)-w >= 0.3*wmx:
        return 5
    # # plt.imshow(im, cmap='gray')
    # # plt.show()

    d = dist(lowest_p, highest_p)
    lower_part_x = round(lowest_p[0] - 0.3*d)
    im2 = np.concatenate( (np.zeros((3, w), dtype=int), im[lower_part_x:][:]), axis=0)
    up2 = highest_point(im2)
    lp2 = lefter_point(im2)
    im2 = draw_line_v(im2, lp2, up2, t=2)
    # plt.imshow(im2, cmap='gray')
    # plt.show()
    num_holes2 = len(get_holes(im2)) -1
    if num_holes2 == 1:
        return 1

    return 2

