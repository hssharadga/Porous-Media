# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:08:50 2020

@author: hssharadga
"""
#import numpy as np
#
##x=[(1,2),(3,4),(5,6)]
##y=x.copy()
##del y [2]
##print (x)
##print (y)
#penetration_length=np.array([[1],[2],[3]])
#
##print (l)

import numpy as np
from numpy import linalg as LA
from math import sqrt, asin,acos, pi, sin, cos, tan, exp
import random
X=[(1,2),(2,3),(3,4),(5,6)]
particle_data = X[:] 
###########################################################################
#b = np.array([2,0.5])
#p2=np.array([4,3])
#p1=np.array([0,0])
#
#
#y = np.divide(p2 - p1, np.linalg.norm(p2 - p1))
#n1 = np.squeeze(np.asarray(p1-b))
#n2 = np.squeeze(np.asarray(y))
#n3 = np.squeeze(np.asarray(b-p2))
#s = np.dot(n1, n2)
#t = np.dot(n3, n2)
#h = np.maximum.reduce([s, t, 0])
#c = np.cross(b - p1, y)
#w = LA.norm(c)
#Dis_center_to_line = sqrt(w**2 + h**2)
#print (Dis_center_to_line)

theta1 = 110*pi/180# pi not 2pi to make sure that the ray dose not cut the sphere which emits energy
#direction cosines of the ray in secondary co-ordinate frame
dir_cos_line = np.array([cos(theta1), sin(theta1)]) 

  
theta=30*pi/180
angle_rotation=90*pi/180-theta    
rot=np.array([[cos(angle_rotation), sin(angle_rotation)],[-sin(angle_rotation), cos(angle_rotation)]])
dir_cos_line_grdframe=np.matmul(rot, dir_cos_line)
print (dir_cos_line_grdframe)   
    
    
  
            