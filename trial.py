# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:43:04 2016

@author: parth
"""

from matplotlib import pyplot as mp
import numpy as np
import math  
'''
decay=0.015
mu=0
sig=1
t=50
def de(t):
    return math.exp(-1*decay*(t))
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
i=0
l=[]
d=[]
while i<t:
    l.append(gaussian(de(i), mu, sig))
    d.append(de(i))
    i+=1
mp.plot(l)
mp.plot(d)
mp.show()
'''

x=np.zeros((m+1,n+1))
elig=np.zeros((m+1,n+1))
w=np.zeros((m+1,n+1))

def main()
    v=np.dot(w.T[time],x[:,time])
    print "w.T[time]",w.T[time],"x",x[:,time],"v[time]",v[time]
main()