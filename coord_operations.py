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

