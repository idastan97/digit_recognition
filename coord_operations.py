import numpy as np
from math import floor, ceil, sqrt
inf = 1000000


def dist_sq(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def dist(p1, p2):
    return sqrt(dist_sq(p1, p2))


def slope(p1, p2):
    if p1[0]-p2[0] == 0:
        return inf
    return (p1[1]-p2[1])/(p1[0]-p2[0])


def mid(p1, p2):
    return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2


def is_orthogonal(p1, p2, q1, q2, tol=0.3):
    return  abs(slope(p1, p2) * slope(q1, q2) + 1.0)/1.0 < tol


def line_shifht_from(q, a):
    return q[1] - a*q[0]


def line_equation(p1, p2):
    a = slope(p1, p2)
    b = line_shifht_from(p1, a)
    return a, b


def get_orthogobal_slope(a):
    if a == 0.0:
        return inf
    return -1/a


def lines_intersection(k, l, a, b):
    x = (b-l)/(k-a)
    y = k*x+l
    return x, y


def orthogonal_line_through(q, k):
    a = get_orthogobal_slope(k)
    b = line_shifht_from(q, a)
    return a, b


def projection_point_to_line(q, k, l):
    a, b = orthogonal_line_through(q, k)
    x, y = lines_intersection(k, l, a, b)
    return x, y


def projection_point_to_segment(q, p1, p2):
    k, l = line_equation(p1, p2)
    return projection_point_to_line(q, k, l)



