# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 01:25:48 2023

@author: aflyn
"""
import numpy as np

class MatrixFactorizer:
    
    def __init__(self,A):
        self.A = A
        u_d = len(A)
        m_d = len(A[0])
        self.U = np.random.rand(u_d,10)
        self.M = np.random.rand(m_d,10)
        
    def gradient(self,stepsize,maxiter,feats=10):
    
        eps = 2.2204e-14 #minimum step size for gradient descent

        loss = 10000
        stepsize = float(stepsize)
        
        A = self.A
        U = self.U
        M = self.M.T
        
        err = np.zeros((A.shape[0],A.shape[1]))
        us = len(A)
        ms = len(A[0])
        crdnl = 1/float(feats)
        
        for i in range(maxiter):
            for j in range(us):
                for k in range(ms):
                    if A[j][k]!=0:
                        err[j][k]=A[j][k]-np.dot(U[j,:],M[:,k])
                        for f in range(feats):    
                            temp_U = U[j][f]+stepsize*(2*err[j][k]*M[f][k]-2/us*U[j][f]) 
                            temp_M = M[f][k]+stepsize*(2*err[j][k]*U[j][f]-2/ms*M[f][k]) 
                            U[j][f] = temp_U
                            M[f][k] = temp_M

            diff = 0
            for j in range(len(A)):
                for k in range(len(A[0])):
                    if A[j][k]!=0:
                        diff += crdnl*(A[j][k]-np.dot(U[j,:],M[:,k]))**2
                        for f in range(feats):
                            diff+=(1/len(A)*(pow(U[j][f],2)+1/len(A[0])*pow(M[f][k],2)))

            if (diff<0.01):
                break
            
        self.A = A
        self.U = U
        self.M = M.T