import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

from image_processing_tools import *
from digit_recognition_tools import *
from coord_operations import *


def recognize(img):
    '''Function to recognize digits on a white paper'''

    # the original image
    print(' - The input image: ')
    plt.imshow(img)
    plt.show()

    print(' - Preprocessing started (it may take several minutes) ...')

    # Preprocessing
    im = np.copy(img)
    h, w, d = im.shape
    
    #   converting to gray image
    im = to_gray(im)
    h, w = im.shape

    #   taking negative
    im = negative(im)

    #   applying sobel edge detection
    # im = sobel(im)
    im = union(sobel(im), dilation(im, mp=thresholding(im, thresh_val(im))))

    #   closing procedure to remove roughness and gaps
    im = closing(im)

    print(' - The result of preprocessing:')
    #   the preprocessed image
    plt.imshow(im, cmap='gray')
    plt.show()
    # plt.imsave('preprocessed.png', im, cmap='gray')
    

    print(' - Segmentation started ...')
    # extracting connected components (digits)
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

    rows = []
    i = 0
    while i < h:
        if np.sum(im[i]) == 0:
            i+=1
        else:
            begin_i = i
            while i<h and np.sum(im[i])>0:
                i+=1
            rows.append((begin_i, i))

    final_nums2 = []

    for i1, i2 in rows:
        # BFS
        nums = []
        im2 = np.copy(im)
        max_pxs = 0
        pxss = []
        cords = []
        for j in range(w):
            for i in range(i1, i2):
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
                    # cords.append(  ( (minx, miny), (maxx, maxy) )  )
                    num = add_frame(get_subarray(num, (minx, miny), (maxx, maxy)), thick=1)
                    nums.append(num)
        # cropping components
        nums2 = []
        for i in range(len(pxss)):
            if pxss[i] <= max_pxs/10:
                continue
            nums2.append(nums[i])
        final_nums2.append(nums2)

    # running recognition for each component
    final_res = []
    for i in range(len(final_nums2)):
        res = []
        for j in range(len(final_nums2[i])):
            res.append(digit_recognize(final_nums2[i][j]))
            print(' - Result of recognition:')
            print(res[-1])
        final_res.append(res)
    return final_res


def digit_recognize(img):

    # Preprocessing
    thickness = 5
    h, w = img.shape
    # resizeing
    dim = 75
    zoom_coef = min(dim/h, dim/w)
    im = zoom_bilinear_interpolation(img, zoom_coef)
    im = add_frame(im)

    h, w = im.shape

    # closing
    im = closing(im)

    # filling defects
    im = defect_filling(im)

    print(' - Segmented digit: ')
    # The result of preprocessing
    show(im)


    print(' - Recognition started ...')
    # count the number of holes
    holes = get_holes(im)
    num_holes = len(holes) - 1

    # 2 holes -> 8
    if num_holes == 2:
        return 8

    # finding highest and lowest points
    lowest_p = lowest_point(im)
    highest_p = highest_point(im)

    # majos/minor axis
    p1, p2 = major_axis(im)
    mj = dist(p1, p2)
    q1, q2 = minor_axis(im, (p1, p2))

    # very thin -> 1
    if dist(q1, q2) < dim*0.1:
        return 1


    if num_holes == 1:
        h1, h2 = major_axis(holes[1])
        # hole is big -> 0
        if dist(h1, h2) > dist(p1, p2)*0.75:
            return 0
        hm = mid(h1, h2)
        # hole at the bottom -> 6/2, at the top -> 9
        if dist(hm, p2) < dist(p1, p2)/2:
            if dist(h1, h2) > dist(q1, q2)/2:
                return 6
            else:
                return 2
        else:
            return 9

    # finding top points
    vertices = []
    tp = top_maxs(im)
    for i in range(len(tp)):
        y1, y2, x = tp[i]
        y = (y1+y2)//2
        if abs(y1-y2) <= thickness:
            if len(vertices) == 0:
                vertices.append((x, y))
            elif abs(vertices[0][0] - x) < mj*0.3 and abs(vertices[0][1] - y) >= thickness:
                vertices.append((x, y))
                break
    if len(vertices) == 2:
        # if 2 top points , then connect them
        four = draw_line_h(im, vertices[0], vertices[1])
        fholes = get_holes(four)
        fnum_holes = len(fholes) - 1
        if fnum_holes == 1:
            return 4

    # find horizontal lines
    hls = horizontal_lines(im, thickness)
    for lp, rp in hls:
        m = mid(lp, rp)
        d = lowest_p[0] - highest_p[0]
        if highest_p[0] + 0.3*d <= m[0] <= lowest_p[0] - 0.3*d:
            return 7

    # find 3 left points
    lps = sorted(list(map(lambda p: ((p[0]+p[1])//2, p[2]), left_maxs(im))))
    d = lowest_p[0] - highest_p[0]
    if (len(lps) == 3 and 
        highest_p[0] + 0.2*d <= lps[1][0] <= lowest_p[0] - 0.2*d and 
        not check_vertical_line(im, lps[0], lps[1]) and 
        not check_vertical_line(im, lps[1], lps[2])):
        return 3

    # analyzing thickness from bottom
    vsum = list(filter(lambda a: a>0, widths(im) ))
    line_lo = 0
    for i in range(-1, -len(vsum)-1, -1):
        if vsum[i] <= thickness:
            line_lo += 1
        else:
            break
    if line_lo > d*0.7:
        return 7
    if line_lo > d*0.35:
        return 1


    # distance from right in upper part vs. distance from left in lower part
    rds = dists_from_right(im)[3:]
    lds = dists_from_left(im)[:-3]
    wmx = w - min(rds) - min(lds)
    mx1 = max(rds[:len(rds)//2])
    mx2 = max(lds[len(lds)//2:])
    if (mx2+mx1)-w >= 0.3*wmx:
        return 5

    # 1 vs. 2: connecting mid point with lower left point
    # hole -> 1, else 2
    d = dist(lowest_p, highest_p)
    lower_part_x = round(lowest_p[0] - 0.3*d)
    im2 = np.concatenate( (np.zeros((3, w), dtype=int), im[lower_part_x:][:]), axis=0)
    up2 = highest_point(im2)
    lp2 = lefter_point(im2)
    im2 = draw_line_v(im2, lp2, up2, t=2)
    # closing
    im2 = closing(im2)
    # filling defects
    im2 = defect_filling(im2)
    holes2 = get_holes(im2)
    num_holes2 = len(holes2)-1
    if num_holes2 >= 1:
        for hl in holes2:
            pm = any_point(hl)
            a, b = line_equation(lp2, up2)
            y = pm[0]*a + b
            if y>=pm[1]:
                return 2
        return 1

    # 2 by default
    return 2

