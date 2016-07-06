from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:16:50 2016

@author: parth
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 00:46:31 2016

@author: parth
"""

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import math 

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
'''Constants '''

alpha=0.01
gamma=0.98
decay=0.985
sigma=0.08
Lambda=0.95

'''Defining matrices'''
m=50
n=500
elig=np.zeros(2*m+1)
w=np.zeros(2*m+1)
delW=np.zeros(2*m+1)
delta=np.zeros(n+1)
r=np.zeros(n+1)
v=np.zeros(n+1)
y_array=[]

'''Functions'''

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/np.sqrt(2*np.pi)

def x_i_t(i,t,shift):
    '''gives the value of x_i at time t'''
    if(t-shift<0):
        y=0
    else:
        y=math.pow(decay,t-shift)
        #print "y",y
    y_array.append(y)
    x_val=gaussian(y,i/m, sigma)*y

    return x_val
    


def plotDelta():
    global delta
    plt.plot(delta)
    plt.show()
    
    
def plotValue():
    global v
    plt.plot(v)
    plt.show()
    
def stimAssign(x,stimulusT):
    '''assigns values to the Stimulus microstimuli'''
    for i in range(0,m):  
        for t in range(0,n+1):   
            x[i][t]=x_i_t(i,t,stimulusT)

def rAssign(x,rewardMag,rewardT):
    '''assigns values to reward microstimuli'''
    r[rewardT]=rewardMag
    for i in range(0,m+1):  
        for t in range(0,n+1):   
            x[i+m][t]=rewardMag*x_i_t(i,t,rewardT)

def runTrial(x):
    elig=np.zeros(2*m+1)
    oldValue=0
    global v
    global w
    
    for time in range(0,n):
        v[time]=max(0,np.dot(w.T,x[:,time]))
            
        delta[time]=r[time]+gamma*v[time]-oldValue
#            print "delta",delta
        delW=alpha*delta[time]*elig
#            print "delW",delW
        w=w+delW
#            print "w", w
        elig=gamma*Lambda*elig+x[:,time]
        oldValue=max(0,np.dot(w.T,x[:,time]))
        
    
def simpleAcq(x,numTrials,rewardMag,rewardT,stimulusT):
    
    stimAssign(x,stimulusT)
    rAssign(x,rewardMag,rewardT)
   
    for trials in range(0,numTrials):
        runTrial(x)        

def rewOmmision(x,numTrials,omTrials,rewardMag,rewardT,stimulusT,):
    
    simpleAcq(x,numTrials,rewardMag,rewardT,stimulusT)
    stimAssign(x,stimulusT)
    rAssign(x,0,rewardT)
    for i in range(0,omTrials):
        runTrial(x)
    

def partialReward(x,numTrials,rewardMag,rewardT,stimulusT,probab):
    
    less=0
    more=0
    absent=np.zeros((2*m+1,n+1))
    stimAssign(x,stimulusT)
    stimAssign(absent,stimulusT)
    rAssign(absent,0,rewardT)
    rAssign(x,rewardMag,rewardT)
    s=np.random.uniform(0,1,numTrials)
#    print s
#    plt.plot(absent[60])
#    plt.plot(x[60)
    for trials in range(0,numTrials):
        if s[trials]<probab:
            runTrial(x)
            r[rewardT]=1
            less+=1
        else:
            runTrial(absent)
            r[rewardT]=0
            more+=1
        
#        plt.plot(delta)
   
    print "less",less,"more",more


def earlyReward(x,numTrials,earlyTrials,rewardMag,rewardT,earlyRewardTime,stimulusT):

    simpleAcq(x,numTrials,rewardMag,rewardT,stimulusT)
    rAssign(x,0,rewardT)    
    rAssign(x,rewardMag,earlyRewardTime)    
    for i in range(0,earlyTrials):
        runTrial(x)
#        if i==0:
#            plotDelta()
#            plotValue()
    
def main():
    numTrials=1000
    rewardMag=1
    x=np.zeros((2*m+1,n+1))
    rewardT=20
    stimulusT=1
    probab=0.25
    omTrials=15
    earlyRewardTime=15
    earlyTrials=15
    simpleAcq(x,numTrials,rewardMag,rewardT,stimulusT)
#    rewOmmision(x,numTrials,omTrials,rewardMag,rewardT,stimulusT)
#    partialReward(x,numTrials,rewardMag,rewardT,stimulusT,probab)
#    earlyReward(x,numTrials,earlyTrials,rewardMag,rewardT,earlyRewardTime,stimulusT)
    plotDelta()
    plotValue()

main()