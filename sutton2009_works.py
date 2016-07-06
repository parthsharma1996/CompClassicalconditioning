# -*- coding: utf-8 -*-
"""
Created on Mon May 30 01:58:13 2016

@author: parth
"""

# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import math 

'''Constants '''
numTrials=3000
n=50
m=20
rewardT=20
stimulusT=2
alpha=0.01
gamma=0.98
decay=0.985
sigma=0.08
Lambda=0.95

'''Defining matrices'''
x=np.zeros((m+1,n+1))
elig=np.zeros((m+1,n+1))
w=np.zeros((m+1,n+1))
delW=np.zeros(n+1)
delta=np.zeros(n+1)
r=np.zeros(n+1)
v=np.zeros(n+1)
y_array=[]

'''Functions'''

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/np.sqrt(2*np.pi)

def x_i_t(i,t,shift):
    '''gives the value of x_i_t'''
    if(t-shift<0):
        y=0
    else:
        y=math.pow(decay,t-shift)
        #print "y",y
    y_array.append(y)
    x_val=gaussian(y,(m-i)/m, sigma)*y
#    if t==0:
#        print "x when i=",i,"t=",t,"is", x_val,"gauss",gaussian(y,i/m, sigma)
    #x_val=stat.norm.pdf(y,i/n, sigma)*y
    #print x_val
    return x_val
    

#print "y+array",y_array

        

for t in range(0,n+1):   
    if(t-rewardT<0):
        r[t]=0
    else:
        r[t]=math.exp(-1*decay*(t-rewardT))

#r[rewardT]=1
        
def main():        
    for i in range(0,m+1):  # let stimulus be at t=20 and reward at t=80
        for t in range(0,n+1):   
            x[i][t]=x_i_t(i,t,stimulusT)
            '''
            print y_array
            plt.plot(y_array)
            plt.show()
            '''
#            
    '''        
    for t in range(0,m+1):   
        plt.plot(x[t])
        plt.show()
    '''        

    for trials in range(0,numTrials):
    
#    for time in range(0,n):
#        v[time]=np.dot(w.T[time],x[:,time])
        for time in range(0,n):
            
            v[time]=np.dot(w.T[time],x[:,time])
            if time==30 and trials==numTrials-1:
                print "w.T[time]",w.T[time],"x",x[:,time],"v[time]",v[time]
            delta[time]=r[time]+gamma*v[time+1]-v[time]
           
            delW=alpha*delta[time]*elig[:,time]
            w[:,time]=w[:,time]+delW
            elig[:,time]=gamma*Lambda*elig[:,time]+x[:,time]
        if trials%200==0: 
            plt.plot(delta)    
#            plt.plot(v)
        #print trials,delta
        #if trials==numTrials-1 : 
#        print trials
            
            
    print "weights",w[0],w[m]
    #print v   
    
   # print delta
    #print w[:,2]
        
        #plt.plot(v)
    
    print delta
    #print v
    plt.show() 

main()