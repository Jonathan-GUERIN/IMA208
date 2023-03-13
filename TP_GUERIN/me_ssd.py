#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:46:43 2022

@author: ckervazo
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import np.linalg as LA
#%%
def computePredictor(r,c,brow,bcol,mvf,ref,cur):
    """
    compute predictor gives the median of the mvf of the blocks :
        - to the left of the current block
        - above the current block
        - upper left of the current block
        
    If such blocks do not exist due to the border effects, they are not taken into account.

    Parameters
    ----------
    See usage in the me_ssd function

    Returns
    -------
    pV : Median of the mvf of the neighboor blocks

    """
    if r < brow and c < bcol:
        pV = initVector(ref,cur)
        
    elif r < brow: # First row
        pV = mvf[r,c-bcol,:]
        
    elif c < bcol: # First column
        pV = mvf[r-brow,c,:]
        
    else: # Inside
        if c >= np.shape(mvf)[1]-bcol: # Last column
            vC = mvf[r-brow,c-bcol,:]
        
        else: # Not the last column
            vC = mvf[r-brow,c+bcol,:]
            
        vA = mvf[r,c-bcol,:]
        vB = mvf[r-brow,c,:]

        temp = np.array([vA, vB, vC]).T

        pV = np.median(temp,axis = 1)
        
    pV = pV.ravel()
    
    return pV

#%%
def initVector(ref,cur):
    """
    Performs an initialization for the first regularizers

    Parameters
    ----------
    ref : np.array
        Reference image.
    cur : np.array
        Reference image.

    Returns
    -------
    pV : np.array (vector of size 2)
        Regularizer for displacement.

    """
    
    
    step = 8
    cont = 4*step
    
    REF = gaussian_filter(ref,1.) # Unclear how to set sigma
    CUR = gaussian_filter(cur,1.)
    
    CUR = CUR[cont+1:(np.shape(CUR)[0]-cont):step,cont+1:(np.shape(CUR)[1]-cont):step]
    SSDMIN = np.inf
    
    pV=np.zeros(2)
    
    for globR in range(-cont,cont):
        for globC in range(-cont,cont):
            RR = REF[cont+1-globR:(cont-globR+np.shape(CUR)[0]*step):step, cont+1-globC:(cont-globC+np.shape(CUR)[1]*step):step]
            SSD = np.sum((RR-CUR)**2)
            
            if SSD<SSDMIN:
                SSDMIN=SSD
                pV[0]=globR
                pV[1]=globC
                
                
    return pV
#%%
def me_ssd(cur, ref, brow, bcol, search, lamb=0):
    """
    ME BMA full search Motion estimation
    mvf, prediction = me_ssd(cur, ref, brow, bcol, search);

    A regularization constraint can be used
    mvf = me(cur, ref, brow, bcol, search, lambda);
    In this case the function minimize SAD(v)+lambda*error(v)
    where error(v) is the difference between the candidate vector v and the
    median of its avalaible neighbors.
 
    Code inspired from the one of Marco Cagnazzo


    Parameters
    ----------
    cur : numpy array
        Current (i.e. second) frame of the video.
    ref : numpy array
        Previous (i.e. first) frame of the video.
    brow : int
        Number of rows in each block.
    bcol : int
        Number of coumns in each block.
    search : int
        Search radius
    lamb : double
        Regularization parameter

    Returns
    -------
    mvf : TYPE
        DESCRIPTION.
    prediction : TYPE
        DESCRIPTION.

    """
    
    extension = search
    
    ref_extended = cv2.copyMakeBorder(ref, extension, extension, extension, extension, cv2.BORDER_REPLICATE) # To avoid border effect
    
    prediction = np.zeros(np.shape(cur));
    lamb *= brow*bcol;
    
    mvf = np.zeros((np.shape(cur)[0],np.shape(cur)[1],2))
    
    # Non-regularized search
    if lamb == 0.:
        for i in range(np.shape(cur)[0]//brow): # for each block in the current image, find the best corresponding block in the reference image
            for j in range(np.shape(cur)[1]//bcol):
                # current block selection
                p = i*brow
                q = j*bcol
                B = cur[p:(p+brow),q:q+bcol] # Block

                # Initialization:
                n, m = p, q
                R = ref_extended[(p+extension):(p+brow+extension),(q+extension):(q+bcol+extension)]
                costMin = LA.norm(B-R)
                
                Rbest = R
                
                # Loop on candidate displacement vectors
                for dcol in range(-search,search,1): # dcol = candidate displacement vector over the columns
                    for drow in range(-search,search,1): # rcol = candidate displacement vector over the rows
                        R = ref_extended[(p-drow+extension):(p-drow+brow+extension),(q-dcol+extension):(q-dcol+bcol+extension)]
                        cost = LA.norm(B-R)
                        
                        if costMin > cost: # Save the results if they are better than the previous ones
                            Rbest = R
                            costMin = cost
                            n = drow
                            m = dcol
                            
                mvf[i:i+brow,j:j+bcol,0] = n # Once the loop is over, save the best row displacement field
                mvf[i:i+brow,j:j+bcol,1] = m # Once the loop is over, save the best column displacement field
                prediction[i:i+brow,j:j+bcol]= Rbest
                
    else: # Regularized search
        for i in range(np.shape(cur)[0]//brow): # for each block in the current image, find the best corresponding block in the reference image
            for j in range(np.shape(cur)[1]//bcol):
                # current block selection
                p = i*brow
                q = j*bcol
                B = cur[p:(p+brow),q:q+bcol] # Block

                # Initializations          
                n, m = p, q
                R = ref_extended[(p+extension):(p+brow+extension),(q+extension):(q+bcol+extension)]
                costMin = LA.norm(B-R)
                
                # Neighbours : pV is the regularization vector. The regularizer must be such that the estimated displacement is not too far away from pV
                pV = computePredictor(i,j,brow,bcol,mvf,ref,cur)
                
                # Loop on candidate vectors
                for dcol in range(-search,search,1): # dcol = candidate displacement vector over the columns
                    for drow in range(-search,search,1): # rcol = candidate displacement vector over the rows
                        R = ref_extended[(p-drow+extension):(p-drow+brow+extension),(q-dcol+extension):(q-dcol+bcol+extension)]
                        cost = LA.norm(B-R) + lamb*((drow-pV[0])**2+(dcol-pV[1])**2) # here, encore the REGULARIZED cost function
                        if costMin > cost: # Save the results if they are better than the previous ones
                            Rbest = R
                            costmin = cost
                            n = drow
                            m = dcol # Save the results if they are better than the previous ones
                
                
                mvf[i:i+brow,j:j+bcol,0]= n # Once the loop is over, save the best row displacement field
                mvf[i:i+brow,j:j+bcol,1]= m # Once the loop is over, save the best column displacement field
                prediction[i:i+brow,j:j+bcol]=Rbest
                
                
    mvf = -mvf # For compatibility with standards
                            
    return mvf, prediction
