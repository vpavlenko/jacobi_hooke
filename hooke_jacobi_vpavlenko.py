#!/usr/bin/env python3

# Vitaly Pavlenko, Fall 2012
# Task #2 for MIPT course on Computational Mathematics

# Fixed points: angular and central ones
# Jacobi method
# Lower-left square m*m has type K_1, other springs has type K_2

N = 10
M = 4
K_1 = 5
K_2 = 2

# N = 4
# M = 2
# K_1 = 5
# K_2 = 2

ILLUSTRATIONS_DIR = '.'
ILLUSTRATIONS_SCALE = 400
ILLUSTRATIONS_PADDING = 40

# Coordinates:
# 10,0 10,1 10,2  ... 10,10
#  ........................
#  1,0  1,1  1,2  ...  1,10
#  0,0  0,1  0,2  ...  0,10


import numpy
from numpy import zeros, linalg, float64, identity, inf
from math import sqrt, log, ceil
import os
import re
import subprocess
import pygame
import sys
# from PySide.QtCore import *
# from PySide.QtGui import *



numpy.set_printoptions(threshold=numpy.nan, linewidth=100000)


fixed = [(0, 0), (0, N / 2), (0, N), (N / 2, 0), (N / 2, N), (N, 0), (N, N / 2), (N - 2, N), (N, N - 2), (N / 2, N / 2)] # (N, N)
fixed_coords = {(i, j): (i / N, j / N) for i, j in fixed}

unfixed = [(i, j) for i in range(0, N + 1) for j in range(0, N + 1) if (i, j) not in fixed]
num_unfixed = (N + 1) ** 2 - len(fixed)
assert num_unfixed == len(unfixed)

flat_index = {(i, j): num for num, (i, j) in enumerate(unfixed)}


def _get_neighbours(i, j):
    return [(x, y) for x, y in [[i - 1, j], [i, j - 1], [i, j + 1], [i + 1, j]] if 0 <= x <= N and 0 <= y <= N]


def _get_k(i, j, x, y):
    if 0 <= i <= M and 0 <= j <= M and 0 <= x <= M and 0 <= y <= M:
        return K_1
    else:
        return K_2


def generate_task():
    '''Return a tuple (A, b) describing the task Ax = b using global coefficients.
    x = (x_0_0, y_0_0, x_0_1, y_0_1, ..., x_0_10, y_0_10, x_1_0, y_1_0, ..., x_10_10, y_10_10)
    For every unfixed node write two equations:
        sum_t k_t (x_we - x_t) = 0
        sum_t k_t (y_we - y_t) = 0
    '''
    A = zeros((2 * num_unfixed, 2 * num_unfixed), dtype=float64)
    b = zeros((2 * num_unfixed), dtype=float64)

    for row, (i, j) in enumerate(unfixed):
        neighbours = _get_neighbours(i, j)
        for x, y in neighbours:
            k = _get_k(i, j, x, y)
            A[2 * row][2 * flat_index[(i, j)]] += k
            A[2 * row + 1][2 * flat_index[(i, j)] + 1] += k
            if (x, y) in fixed:
                b[2 * row] += k * fixed_coords[(x, y)][0]
                b[2 * row + 1] += k * fixed_coords[(x, y)][1]
            else:
                A[2 * row][2 * flat_index[(x, y)]] -= k
                A[2 * row + 1][2 * flat_index[(x, y)] + 1] -= k
    
    return A, b


def extract_diagonal_matrix(A):
    '''Return a matrix D whose diagonal elements are equal to correspondent of A
    and non-diagonal ones are zeros.'''
    d1, d2 = A.shape
    assert d1 == d2
    D = zeros((d1, d1), dtype=float64)
    for i in range(d1):
        D[i][i] = A[i][i]
    return D    


def invert_diagonal_matrix(A):
    D = A.copy()
    for i in range(A.shape[0]):
        D[i][i] = 1. / A[i][i]
    return D


def transform_to_jacobi_problem(A, b):
    '''Return tuple (B, g) describing the task x = Bx + g.
    '''
    D = extract_diagonal_matrix(A)
    D_1 = invert_diagonal_matrix(D)
    E = identity(A.shape[0])
    B = E - D_1.dot(A)
    g = D_1.dot(b)
    return B, g


def generate_initial_approximation():
    x = zeros((2 * num_unfixed), dtype=float64)
    for num, (i, j) in enumerate(unfixed):
        x[2 * num] = i / N
        x[2 * num + 1] = j / N
    return x


class Drawer:

    def __init__(self, filename):
        self.filename = filename
        self.border_size = ILLUSTRATIONS_SCALE + 2 * ILLUSTRATIONS_PADDING   

    def save(self):
        raise NotImplementedError()

    def draw_line(self):
        raise NotImplementedError()

    def draw_point(self):
        raise NotImplementedError()

    @classmethod
    def resolve_color(cls, color):
        return {'white': (255, 255, 255),
                'blue':  (0, 0, 255),
                'green': (0, 255, 0),
                'red': (255, 0, 0),}[color] 

    @classmethod
    def transform_coords(cls, coords):
        res = [int(round(ILLUSTRATIONS_PADDING + x * ILLUSTRATIONS_SCALE)) for x in coords]
        return res


class QtDrawer(Drawer):

    def __init__(self, filename):
        Drawer.__init__(self, filename)
        subprocess.call('convert -size {0}x{0} xc:white empty.png'.format(self.border_size))
        self.app = QApplication([])
        self.img = QImage('empty.png')
        self.canvas = QPainter(self.img)

    def save(self):
        self.canvas.end()
        self.img.save(self.filename)

    def draw_line(self, from_coords, to_coords, color):
        canvas.setPen(QColor(*Drawer.resolve_color(color)))
        canvas.drawLine(QPoint(*Drawer.transform_coords(from_coords)), QPoint(*Drawer.transform_coords(to_coords)))

    def draw_point(self, coords, color):
        canvas.setBrush(QColor(*Drawer.resolve_color(color)))
        # canvas.drawEllipse(coords[0], coords[1], 5, 5)        


class PyGameDrawer(Drawer):

    def __init__(self, filename):
        Drawer.__init__(self, filename)
        self.screen = pygame.display.set_mode((self.border_size, self.border_size))

    def draw_line(self, from_coords, to_coords, color):
        pygame.draw.lines(self.screen, Drawer.resolve_color(color), False, 
                [Drawer.transform_coords(from_coords), Drawer.transform_coords(to_coords)], 2)

    def draw_point(self, coords, color):
        pygame.draw.circle(self.screen, Drawer.resolve_color(color), Drawer.transform_coords(coords), 4, 0)

    def save(self):
        pygame.image.save(self.screen, self.filename)



def visualize(x_vec, picture_number):
    drawer = PyGameDrawer('{0}.png'.format(picture_number))

    def resolve_coords(i, j):
        if (i, j) in fixed:
            return fixed_coords[(i, j)]
        else:
            return x_vec[2 * flat_index[(i, j)]:2 * flat_index[(i, j)] + 2]


    for i in range(0, N + 1):
        for j in range(0, N + 1):
            for x, y in _get_neighbours(i, j):
                drawer.draw_line(resolve_coords(i, j), resolve_coords(x, y), 
                        'blue' if _get_k(i, j, x, y) == K_1 else 'green')

    for i in range(0, N + 1):
        for j in range(0, N + 1):
            if (i, j) in fixed:
                drawer.draw_point(resolve_coords(i, j), 'red')
            else:
                drawer.draw_point(resolve_coords(i, j), 'white')

    drawer.save()


def calc_discrepancy(x, print_=False):
    discr = linalg.norm(A.dot(x) - b, 2)
    if print_:
        print('Discrepancy: ||Ax - b||_2 = {0}'.format(discr))
    return discr


if __name__ == '__main__':
    discr = []

    try:
        os.mkdir(ILLUSTRATIONS_DIR)
    except OSError:
        pass
    os.chdir(ILLUSTRATIONS_DIR)

    for filename in os.listdir('.'):
        if re.match(r"[0-9]+\.png", filename):
            os.remove(filename)

    A, b = generate_task()
    print('We need to solve the problem Ax = B where')
    print('A = ')
    print(A)
    print()
    print('b = ')
    print(b)
    print()
    x = generate_initial_approximation()
    print('We start from the initial approximation x_0 = ')
    print(x)
    discr.append(calc_discrepancy(x, True))
    print()
    print('All points lie in a square [0, 1]*[0, 1], so ||x - x_0||_2 <= sqrt(len(x_0) * 1) =', sqrt(len(x)))
    visualize(x, 0)
    B, g = transform_to_jacobi_problem(A, b)
    print('In order to apply Jacobi Iterative Method "x = Bx + g" we get matrix B and vector g where')
    print('B = ')
    print(B)
    print()
    print('g = ')
    print(g)
    print()
    B_norm = linalg.norm(B, 2)
    print('||B||_2 = {0}'.format(B_norm))
    print()
    print('As long as ||B||_2 < 1, the Jacobi Method converges')
    print()
    num_iterations = ceil(log(1e-8 / sqrt(len(x)), linalg.norm(B, 2)))
    print('We need to obtain ||x_k - x|| <= 1e-8. As long as ||x_k - x|| <= ||B||^k * ||x - x_0||, k := log(1e-8 / ||x - x_0||, ||B||) =',
            num_iterations, 'gives necessary precision of the answer')
    print()
    old_x = None
    for i in range(1, num_iterations + 1):
        x = B.dot(x) + g
        # print(x)
        discr.append(calc_discrepancy(x))
        if i <= 10 or (i < 100 and i % 10 == 0) or i % 100 == 0 or i == num_iterations:
            visualize(x, i)
            print('After {0} iterations x ='.format(i))
            print(x)
            if old_x is not None and (x == old_x).all():
                print('(x_{0} == x_{1} so we have already obtained the solution)'.format(i, old_i))
            calc_discrepancy(x, True)
            print()                
            old_x, old_i = x, i
    print('Final solution as a table of (x, y)-coords:')
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            coords = fixed_coords[(i, j)] if (i, j) in fixed else (x[2 * flat_index[(i, j)]], x[2 * flat_index[(i, j)] + 1])
            print('({0:.8}, {1:.8})'.format(*coords), end='\t')
        print()
    print('Accumulated data for discrepancy:')
    print('c(' + ', '.join(['{0:.2}'.format(i) for i in discr]) + ')')
