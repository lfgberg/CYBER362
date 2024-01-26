# -*- coding: utf-8 -*-
"""
CYBER362 Lab 2 Section A
@author: lfg5289
"""
import numpy as np

# 1.4.2 - Creating Arrays
# Manual - 1D
a = np.array([0, 1, 2, 3])
print(a)

# Manual - 2D
b = np.array([[0, 1, 2], [3, 4, 5]])
print(b)

# Manual - 3D
c = np.array([[[1], [2]], [[3], [4]]])
print(c)

# Random
d = np.random.rand(5)
print(d)

# 1.4.3 - Datatypes
e = np.array([0, 1, 2, 3], dtype=float)
print(e)
