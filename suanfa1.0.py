# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:10:11 2017

@author: Mika
"""

#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
try:
    import numpy
    import numpy.linalg as LA
except:
    print "This implementation requires the numpy module."
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    I     : a matrix to be added,dimension N x C
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    S     : an initial matrix of dimension C x K
    K   : the number of latent features for R and I
    R=P * Q;
    I=P * S；
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q and S
"""    
def matrix_factorization(R, I, P, Q, S, K, steps=5000, lamda=0.6, alpha=0.0002, beta=0.02):
    Q = Q.T
    if S!=None:
        S = S.T        
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    if I!=None:#have transfer learning#
                        for x in xrange(len(I[i])):
                            if I[i][x] >0:
                                eix= I[i][x] - numpy.dot(P[i,:],S[:,j])
                                for k in xrange(K):
                                    P[i][k] = P[i][k] + alpha * (2 * lamda * eij * Q[k][j] + 2 * (1 - lamda) * eix * S[k][x] - beta * P[i][k])
                                    Q[k][j] = Q[k][j] + alpha * (2 * lamda * eij * P[i][k] - beta * Q[k][j])
                                    S[k][x] = S[k][x] + alpha * (2 * (1-lamda) * eix * P[i][k] - beta * S[k][x])          
                    else:#no transfer learning#
                        for k in xrange(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e_1=0
        e_2=0
        reg=0 
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e_1 = e_1 + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        reg = reg + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
            
        if I !=None:
            for x in xrange(len(I[i])):
                if I[i][x] > 0:
                    e_2 = e_2 + pow(I[i][x] - numpy.dot(P[i,:],S[:,x]), 2)
                    for k in xrange(K):
                        reg = reg + (beta/2) * ( pow(S[k][x],2) )  
        
            e=lamda*e_1+(1-lamda)*e_2+reg
        else:
            e=e_1+reg
        if e < 0.001:
            break
    print ('the loss is:')
    print e
    return P, Q.T

def RMSE(X, Y):
    S=pow(X-Y,2)
    S=S.mean()
    return numpy.sqrt(S.mean)
###############################################################################

if __name__ == "__main__":
    filename='E:/MSBD5001/test/app_matrix.csv'
    from numpy import genfromtxt
    my_data=genfromtxt(filename,delimiter=',')
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]
    R1 = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]
    
#    my_data=numpy.delete(my_data,0,1)
#    my_data=numpy.array(my_data)
#    print my_data[0]
#    #print type(my_data)
#    #print my_data.shape
#    
#    R = numpy.array(my_data)
#    R=R[:5]
    N = len(R)#the row of R#

    M = len(R[1])#the colume of user*article#
#    print M
    K = 5
    C = len(R1[1])#the colume of user*apps#

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)
    S = numpy.random.rand(C,K)
    nP, nQ= matrix_factorization(R,None, P, Q, None, K)

    nR=numpy.dot(nP,nQ.T)
    print nR
    
    