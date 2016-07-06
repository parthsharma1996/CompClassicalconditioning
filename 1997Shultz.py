# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:24:07 2016

@author: parth
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor




This is a temporary script file.
"""
numTrials=100
n=20
rewardT=18
stimulusT=2
alpha=0.4
gamma=0.999

import numpy as np
import matplotlib.pyplot as plt

x=np.zeros((n+1,n+1))
w=np.zeros(n+1)
delW=np.zeros(n+1)
delta=np.zeros(n+1)
r=np.zeros(n+1)

for i in range(0,n-stimulusT+1):  # let stimulus be at t=20 and reward at t=80
    x[i][i+stimulusT]=1

r[rewardT]=1   
trials=0

for trials in range(0,numTrials):
    time=0
    
    delW=np.dot(delta,(x.T))
    delW=delW*alpha
    w=w+delW
    v=np.dot(w,x)
    for time in range(0,n):
        delta[time]=r[time]+gamma*v[time+1]-v[time]
   # print delta
    if trials%10==0: 
        print trials
        plt.plot(delta)

plt.show()