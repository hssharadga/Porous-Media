# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:27:18 2020

@author: hssharadga
"""
import numpy as np
from numpy import linalg as LA
from math import sqrt, asin,acos, pi, sin, cos, exp,tan,atan
import random
import csv 

import timeit

start = timeit.default_timer()


########### import
#Code to import the circels centers  from the results previous code (2D packing),  a csv file
# =============================================================================
f = open('Circels.csv','r')# Circels is the name of the file
reader = csv.reader(f)
data = []
for row in reader:
    data.append(row)
data = np.array(data,float)
# =============================================================================
xx1 = data.tolist()

#xx1 = data



## inputs
# note: update sp_x

Num=1# number of iterations
D = 100 #sphere diameter(cm)
eta = 0.4 #absorbtivity
nta = 2000 #refractive index
ex = 0.3 #extinction coefficient
nta_air = 1.0003
ymax=2483/2
ymin=0

xmax=2483
xmin=0
#y1=xmax/2*tan(30*pi/180)# the source of light; distance to the bottom of the porous
y1=-D # it should be closed or less than the distance of travling to hit the sphere of first raw
dim = np.array([[xmin,ymin],[xmax,ymax]]) #bed dimentions start to end in the groung frame(cm)
l=20*D# the length of ray is eougth to hit at leasat one sphere
##############################################3



X = xx1 # xx1 is the data set with the origins of the all the particles in ground frame
particle_data = X[:] 

#X = []#points that a all the rays hit
E1 = []  # energy that is transmitted through the bed
E2 = [] # energy being reflected back
E3 = [] # energy going out through the walls
for i in range(Num):
    
    print ('Number of iterations :', i,'out of', Num)
#initial ray
    # random initial point for sending the ray in the particle bed 
    sp_x = xmax/2 +(0.5-((random.random())))* 0.6*xmax
#    print (sp_x)
#    sp_x=xmax/2
    p1= np.array([sp_x,y1])# the source of light; firsit point
    ## 
    dir_cos_ini = np.array([0,1])# the firsit direction will be with angel 90
    
    

    #pts is a list to add all the points being hit by the ray and dir_cos is the list to save direction cosines
    pts = np.array([0,0])
    pts = np.vstack((pts,p1))
    all_point=pts[:]
    dir_cos = np.array([0,0])
    dir_cos = np.vstack((dir_cos,dir_cos_ini))
    sphere_centers_hit = np.array([[0,0],[0,0]]) # all the spheres being hit by the ray
    I = 1
    ii = 1 # inicializing for use in the code
    
    while I>  1E-10:
        X = xx1 # list of all particles
#finding the sphere hit by this ray given the initial point and the direction cosines
        p1= np.array([pts[ii][0],pts[ii][1]]) #initial point outside the bed
        p2= np.array([pts[ii][0]+dir_cos[ii][0]*l,pts[ii][1]+dir_cos[ii][1]*l]) # second point taken randomly on the ray at a distance of 10cm from initial point
        p1 = np.reshape(p1,(1,2))# reshape to make it compatible for further calculations
        p2 = np.reshape(p2,(1,2))
        sphere_hitting = np.array([[0,0],[0,0]]) # list being inicialized at each step to save all the spheres being hit by this ray so that later we can calculate the the first(nearest) hit point and sphere 
        particle_data_new = X[:] # data from which the sphere hit from lst iteration will be deleted to have proper further calculations
        if ii >1: # supressing first iteration
            index1,index2= np.where(X == C)  # finding the index of last hit sphere
            del particle_data_new[index1[1]] # deleting the last hit sphere
       
        
        # code for finding all the spheres being hit by finding the distance of the centers from this ray and the saving the particles with distance less than D/2(radius)
        for a in particle_data_new[0:]:
            b = np.array(a)
            y = np.divide(p2 - p1, np.linalg.norm(p2 - p1))
            n1 = np.squeeze(np.asarray(p1-b))
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
#        print (sphere_hitting)
        #checking if the ray is not hitting any particles then it has left the bed and thus checking which boundary did the ray leave
        if len(pts)>2:
#            p3 = np.array([pts[ii][0]+dir_cos[ii][0]*D,pts[ii][1]+dir_cos[ii][1]*D])
            if len(sphere_hitting)<3:
                if p2[0][1]<ymin: # reflection
                    E2.append(I)
                if p2[0][1]>ymax: # transmitted
                    E1.append(I)
                if p2[0][0]<xmin or p2[0][0]>xmax: # passing through the side boundries
                    E3.append(I)
                break
# code to find the closest sphere hit by the ray       
        y = 10000# random initialization         
        for b in sphere_hitting[2:]:
            dis = sqrt((pts[ii][0] - b[0])**2+(pts[ii][1] - b[1])**2)# the closet sphere to the emission point
            # not to the final point becuse the final point is an assumption based on length of ray which is not predictable
            
            if dis< y:
                y = dis
                sphere_being_hit_center = b
       
        C = sphere_being_hit_center # saving the sphere being hit in a variable c 
#        print (C)
        sphere_centers_hit = np.vstack((sphere_centers_hit,C))
#Code for finding the points on this spphere that are hit by the ray and then selecting the first hit point
        p1 = np.squeeze(np.asarray(p1))
        p2 = np.squeeze(np.asarray(p2))
        
        u = (p2[0] - p1[0])**2+(p2[1] - p1[1])**2
        v = 2*(((p2[0] - p1[0])*(-sphere_being_hit_center[0]+p1[0])) + ((p2[1] - p1[1])*(-sphere_being_hit_center[1]+p1[1])))
        w = (sphere_being_hit_center[0]-p1[0])**2 + (sphere_being_hit_center[1]-p1[1])**2-(D/2)**2
        t1 = (-v + sqrt((v**2)-4*u*w))/(2*u)
        t2 = (-v - sqrt((v**2)-4*u*w))/(2*u)
        ep1 = np.array([p1[0]+(p2[0]-p1[0])*t1,p1[1]+(p2[1]-p1[1])*t1])
        ep2 = np.array([p1[0]+(p2[0]-p1[0])*t2,p1[1]+(p2[1]-p1[1])*t2])
        dis1 = sqrt((p1[0]-ep1[0])**2 + (p1[1]-ep1[1])**2)
        dis2 = sqrt((p1[0]-ep2[0])**2 + (p1[1]-ep2[1])**2)
        if (dis1 <= dis2):
            ep = ep1
        else:
            ep = ep2
#Code for finding whether tha ray is getting refracted or reflected
        if ep[1]< ymax and ep[1]>ymin:
            normal = np.array([ep[0]-C[0],ep[1]-C[1]])
            norm = LA.norm(normal)
            normal_dir_cos = np.array([normal[0]/norm, normal[1]/norm])
            phi = acos(np.dot(-normal_dir_cos,dir_cos[ii])) ##cos(theta_incident)=N.I since N= final - inital
            #that mean it exit the sphere at the hitting point but according to the figure of the refernce; N cut the sphere completely
            phi2 = asin(nta_air*sin(phi)/nta)
            rho_parallel = (nta*cos(phi)-nta_air*cos(phi2))/(nta*cos(phi)+nta_air*cos(phi2))
            rho_perpendicular = (nta_air*cos(phi2)-nta*cos(phi))/(nta_air*cos(phi2)+nta*cos(phi))
            
            trans_parallel = (2*sin(phi2)*cos(phi))/(sin(phi+phi2)*cos(phi-phi2))
            trans_perpendicular = (2*sin(phi2)*cos(phi))/(sin(phi+phi2))
            
            rho_avg = (rho_parallel**2 + rho_perpendicular**2)/2
            trans_avg = (trans_parallel**2 + trans_perpendicular**2)/2    
            rand1 = np.random.random(1)[0]
            
            
            
            
            if rand1< rho_avg: # spectral reflection
                dot_prod = np.dot(normal_dir_cos,dir_cos[ii])
                
                ## R= I - 2 *(N.I) N
                ## here N if postive or negatvie it dose not matter becuase the N twice
                reflected_dir_cos1 = np.array([(dir_cos[ii][0] - 2*normal_dir_cos[0]*dot_prod), (dir_cos[ii][1] - 2*normal_dir_cos[1]*dot_prod)])
                norm1 = LA.norm(reflected_dir_cos1)
                reflected_dir_cos = np.array([reflected_dir_cos1[0]/norm1,reflected_dir_cos1[1]/norm1])
                angel_inci_reflect = acos(np.dot(reflected_dir_cos,normal_dir_cos))
                pts = np.vstack((pts,ep))
                all_point=np.vstack((all_point,ep))
                
                dir_cos = np.vstack((dir_cos,reflected_dir_cos))
                ii = ii+1
                I = I - I*eta
                
                
    
            else: #refraction
                n = nta_air/nta
                cphi = np.dot(-normal_dir_cos,dir_cos[ii])
                phi = acos(np.dot(-normal_dir_cos,dir_cos[ii]))
                phi2 = asin(n*sin(phi))
                c1 = np.dot(-normal_dir_cos,dir_cos[ii])
                c2 = 1 - ((n**2)*(1-cphi**2))
                # refracted direction cosines
                refracted_dir_cos1 = np.array([(n)*(dir_cos[ii][0]) +(n*cphi-sqrt(c2))*normal_dir_cos[0], (n)*(dir_cos[ii][1]) +(n*cphi-sqrt(c2))*normal_dir_cos[1]])
                norm1 = LA.norm(refracted_dir_cos1)
                refracted_dir_cos = np.array([refracted_dir_cos1[0]/norm1,refracted_dir_cos1[1]/norm1])
                 # find the final point after the ray has refracted and travelled through the sphere
                p1 = ep
                p2 = np.array([p1[0]+refracted_dir_cos[0]*1,p1[1]+refracted_dir_cos[1]*1])
                
                
                u = (p2[0] - p1[0])**2+(p2[1] - p1[1])**2
                v = -2*((p2[0] - p1[0])*(C[0]-ep[0])) + -2*((p2[1] - p1[1])*(C[1]-ep[1]))
                w = (C[0]-ep[0])**2+(C[1]-ep[1])**2-(D/2)**2
                t1 = (-v + sqrt(v**2-(4*u*w)))/(2*u)
                ep2 = np.array([ep[0]+ t1*(p2[0] - p1[0]), ep[1]+ t1*(p2[1] - p1[1])])
                s = sqrt((ep2[0]-ep[0])**2 + (ep2[1]-ep[1])**2)
                normal2 = np.array([ep2[0]-C[0],ep2[1]-C[1]])
                norm2 = LA.norm(normal2)
                normal_dir_cos2 = np.array([normal2[0]/norm2, normal2[1]/norm2])
                b1 = nta/nta_air
                phi3 = acos(np.dot(normal_dir_cos2,refracted_dir_cos))
                phi4 = asin(b1*sin(phi3))
                #print(phi4)                         
                c3 = np.dot(normal_dir_cos2,refracted_dir_cos)
                c4 = 1 - ((b1**2)*(1-c3**2))
                refracted_dir_cos2 = np.array([(b1)*(refracted_dir_cos[0]) - (b1*c3-sqrt(c4))*normal_dir_cos2[0], (b1)*(refracted_dir_cos[1]) - (b1*c3-sqrt(c4))*normal_dir_cos2[1]])
                norm3 = LA.norm(refracted_dir_cos2)
                refracted_dir_cos3 = np.array([refracted_dir_cos2[0]/norm3,refracted_dir_cos2[1]/norm3])
                #print(acos(np.dot(normal_dir_cos2,refracted_dir_cos3)))
                pts = np.vstack((pts,ep2))
                dir_cos = np.vstack((dir_cos,refracted_dir_cos3))
                ii = ii+1
                I = I - I*eta # need to decide how much enerfy gets lost
                
                all_point1=np.vstack((all_point,ep))
                all_point=np.vstack((all_point1,ep2))
        
        elif ep[1]< ymin:
            I = I
            E2.append(I)
            break
        else:
            E1.append(I)
            break                
print ('Transmitted', sum(E1))
print ('reflected',sum(E2)) 
print ('escape',sum(E3))

stop = timeit.default_timer()

print('Time: ', stop - start)       