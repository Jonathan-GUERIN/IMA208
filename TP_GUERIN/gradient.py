#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:06:43 2022

@author: ckervazo
"""
import numpy as np
from scipy import ndimage

def gradient(M,stepX=1.,stepY=1.):
    
    # Computes the gradient of an image, over the rows and the column directions. StepY is the assumed gap between the rows and StepX is the assumed gap between the columns
    
    gy = np.zeros((M.shape))
    gx = np.zeros((M.shape))
    for i in range(M.shape[0]-1):
        for j in range(M.shape[1]-1):
            gy[i,j] = M[i,j] - M[i,j+1] / stepY
            gx[i,j] = M[i,j] - M[i+1,j] / stepX
    for i in range(M.shape[0]):
        gy[i,M.shape[1]-1] = 0
        gx[i,M.shape[1]-1] = 0
    for j in range(M.shape[1]):
        gy[M.shape[0]-1,j] = 0
        gx[M.shape[0]-1,j] = 0
    
    
    return gx,gy
