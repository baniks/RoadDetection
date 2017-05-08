#!/usr/bin/python
#######################################################################
#   File name: edge.py
#   Author: Soubarna Banik
#   Description: Adaptation of Felzenszwalb's segmentation algorithm.
#                Contains class Edge
#######################################################################


class Edge:
    """
    Class edge
    a: pixel number
    b: neighboring pixel number
    w: edge weight
    """
    def __init__(self, a, b, w):
        self.a = a
        self.b = b
        self.w = w
