# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:26:32 2020

@author: hssharadga
"""
### Method

# we pick a ranodm sphere using random angle then we convert those angle to a point on the sphere surface
# this point is called the emission point then we emit random ray using a random angle then we convert this direction to the ground frame using rotation matrix
# then we find the sphere being hiited be calcuating the distance and then find the closeest sphere
# the closest sphere is hitted at two point, we find the closeest one

##input 

D = 50  # diameter of the particle
Num=10000 # number of iteration
xlow=100
xhigh=2300
ylow=100
yhigh=2300/2
#################



import numpy as np
from numpy import linalg as LA
from math import sqrt, asin, pi, sin, cos
import random
import csv




#Code to import sphere center from the results of 2D packing,  a csv file
# =============================================================================
f = open('Circels.csv','r')# Circels is the name of the file
reader = csv.reader(f)
data = []
for row in reader:
    data.append(row)
data = np.array(data,float)
# =============================================================================
X = data.tolist()  #Converting data from a Numpy array to a list
penetration_len = [] #Initializing the list to save the value of penetration length at each iteration


trace=0 # the number of the calucalted penetration length 

for i in range(Num): # number of iterations
    print ("Number of iterations", i, 'out of ', Num)
#If the selected sphere is near the walls then find a new sphere  
    sphere_center_data_new = X #new list to iterate over in the code
    while True: 
        g = np.random.randint(1,len(X)-1) # randomly choose a sphere from the list
        
        main_sphere_center = X[g]
#If the selected sphere is near the walls the find a new sphere  
        if main_sphere_center[0]>xlow or main_sphere_center[0]<2300 or main_sphere_center[1]>ylow or main_sphere_center[1]<yhigh :
            break
#remove the selected sphere from this list
#    print (g)
#    print (len(X))
#    print (main_sphere_center)
    new=sphere_center_data_new.copy()

    del new[g]
## we delete from the array because when we calcualte the distance
    ## this sphere should not be taken because the distance between sphere and it seld is zero
    
#Code to get the direction cosines of the emitted ray in the ground frame making sure that this ray doesn't penetrate any sphere
# Direction cosines of the ground frame
    i1 = np.array([1,0,0])
    j1 = np.array([0,1,0])
    k1 = np.array([0,0,1])
    
#radom point on the selected sphere
    
    R_theta = np.random.random(1)[0]
    theta = 2*pi*R_theta # 2pi because it can be in all direction
    
#Direction cosines of the random point on the sphere
    dir_cos_main = np.array([cos(theta),sin(theta)])
    #co-ordinates of the emission point
    emission_point = np.array([main_sphere_center[0]+cos(theta)*(D/2), main_sphere_center[1]+sin(theta)*(D/2)])
    ## we mutliply by D/2 to find the point on the sphere surface because the cos and sin is the direction
    ## we add the sphere center to convert it in the ground frame 
    
    
    
    
# we emit a ray from the emissiom point in a random direction
   
    R_theta1 = np.random.random(1)[0]
    theta1 = pi*R_theta1# pi not 2pi to make sure that the ray dose not cut the sphere which emits energy
#direction cosines of the ray in secondary co-ordinate frame
    dir_cos_line = np.array([cos(theta1), sin(theta1)])
    
    
    
#    # Rotation matrix to get the direction of emitted ray in the ground frame
#    #K2 is the rotation axis
#    k2_ = np.array([emission_point[0],emission_point[1]])
#    norm1 =LA.norm(k2_)
#    k2 = k2_/norm1
#    norm2 = np.cross(k2,k1)
#    i2 = (np.cross(k2,k1))/LA.norm(norm2)
#    j2 = np.cross(k2,i2)
##rotation matrix to get direction cosines in the ground frame
#    rot = np.array([[i2],[j2],[k2]])
#    rot = rot.transpose()
#    
##direction cosines in the ground frame
#    dir_cos_line_grdframe = dir_cos_line
#    
##    dir_cos_line_grdframe = np.matmul(rot,dir_cos_line)
    # convert the direction of ray to the ground frame
    dir_cos_line_grdframe=np.array([cos(theta1-(90*(pi/180)-theta)),sin(theta1-(90*(pi/180)-theta))])
    
    
    
#initializing the array to save the final result
    sphere_hitting = np.array([[0,0],[0,0]])
#calculation of the distance of all the spheres in the list to the emitted ray and checking the spheres that are being hit by the ray
    p1= np.array([emission_point[0],emission_point[1]])
    p2= np.array([emission_point[0]+dir_cos_line_grdframe[0]*10,emission_point[1]+dir_cos_line_grdframe[1]*10])
    ## p2 is the final point on the emiited ray, P1 is the firsit point
    
    p1 = np.reshape(p1,(1,2))
    p2 = np.reshape(p2,(1,2))
 # To find the distance between line and point (center of circle)    
    for a in new:
        b = np.array(a)
        y = np.divide(p2 - p1, np.linalg.norm(p2 - p1)) 
        n1_ = np.squeeze(np.asarray(p1-b))
        n1 = np.array([n1_[0] - b[0],n1_[1] - b[1]])
        n2 = np.squeeze(np.asarray(y))
        n3 = np.squeeze(np.asarray(b-p2))
        s = np.dot(n1, n2)
        t = np.dot(n3, n2)
        h = np.maximum.reduce([s, t, 0])
        c = np.cross(b - p1, y)
        w = LA.norm(c)
        Dis_center_to_line = sqrt(w**2 + h**2)
        if Dis_center_to_line < D/2:
            sphere_hitting= np.vstack((sphere_hitting,a))
            
    xx = 100000   #random initialization
    if len(sphere_hitting)==2: # If no sphere is hitting the ray then move to the next iteration
        continue
    trace=trace+1
#Finding the centers of the closest sphere hitting the ray
        
    for a in sphere_hitting[2:]:
        dis = sqrt((main_sphere_center[0] - a[0])**2 + (main_sphere_center[1] - a[1])**2 )
        if dis< xx:
            xx = dis
            closest_sphere = a
            
            
#calculating the points at which the line or ray cut the sphere           
    p1 = np.squeeze(np.asarray(p1))
    p2 = np.squeeze(np.asarray(p2))
    
    
    u = (p2[0] - p1[0])**2+(p2[1] - p1[1])**2 
    ## it can be direction (dir_cos_line_grdframe) instead of P2- P1
    ## in that case ep1 and ep2 we should use the direction instead of P2-P1
    v = -2*((p2[0] - p1[0])*(closest_sphere[0]-p1[0])) + -2*((p2[1] - p1[1])*(closest_sphere[1]-p1[1]))
    w = (closest_sphere[0]-emission_point[0])**2+(closest_sphere[1]-emission_point[1])**2-(D/2)**2
    t1 = (-v + sqrt(v**2-4*u*w))/(2*u)
    t2 = (-v - sqrt(v**2-4*u*w))/(2*u)
    
    # two point on the sphere ep1, ep2. We have to check which one is the closet
    
    ep1 = np.array([emission_point[0]+(p2[0]-p1[0])*t1,emission_point[1]+(p2[1]-p1[1])*t1])
    ep2 = np.array([emission_point[0]+(p2[0] - p1[0])*t2,emission_point[1]+(p2[1] - p1[1])*t2])
    
#the distance between the hitting points and the emission point to get the lenght travelled by the ray before hitting
    penetration_length1 = sqrt((emission_point[0]-ep1[0])**2 + (emission_point[1]-ep1[1])**2) 
    penetration_length2 = sqrt((emission_point[0]-ep2[0])**2 + (emission_point[1]-ep2[1])**2)
    if (penetration_length1 <= penetration_length2):
        penetration_length = penetration_length1
    else:
        penetration_length = penetration_length2
#adding the calculated penetration length to the list
    penetration_len.append(penetration_length)
    
penetration_len=np.asarray(penetration_len)
penetration_len=np.reshape(penetration_len,(trace,1))

average=np.sum(penetration_len)/trace
print ('average = ', average)
##Saving the results in a file
with open('penetration_length.csv', "w", newline = '') as output: ## penetration_length is the name of excel file that will be produced
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(penetration_len) ## penetration_len is waht I want the python to write out
    ## penetration_len should be array not list
    
    
    
    