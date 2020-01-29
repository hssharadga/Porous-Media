# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:27:44 2020

@author: hssharadga
"""
##Method 
## after we calcualte the penetration length, we start from the origin
# we emit ray in random direction with length of penetration length (we emit ray from (0,0))
# then we find a random sphere the ray hits, the sphere center will be choosen randomly, but also we have to 
# make sure that the ray dose not hit on any exist sphere by calcualting the distances
# then we calulate the new direction based on if it reflection or refraction
# after we calcuale the the direction we emit the ray with length of penetration length again
# and keep tracing 

# the origin is (0,0), the dimission is infinity in x but determined for y

import numpy as np
from numpy import linalg as LA
from math import sqrt, asin,acos, pi, sin, cos, tan, exp,atan
import random
import timeit
import csv
start = timeit.default_timer()





#Code to import the pentration length  from the results previous code,  a csv file
# =============================================================================
f = open('penetration_length.csv','r')# Circels is the name of the file
reader = csv.reader(f)
data = []
for row in reader:
    data.append(row)
data = np.array(data,float)
# =============================================================================
#X = data.tolist()  #Converting data from a Numpy array to a list
penetration_len = data





# inputs
Num=1 # number of iterations
D = 100 #sphere diameter(cm)
eta = 0.4 #absorbtivity
nta = 2 #refractive index
ex = 0.3 #extinction coefficient
nta_air = 1.0003
A = np.array([2438,2438/2]) #bed dimentions(cm)
gab=10

E1 = [] #save intencities that gets transmitted
E2 = [] # saves intencities that are reflected




for i in range(Num): #number of iterations
#ground frame
    print ('Number of iterations = ',i)
    
    
    o = np.array([0,0])# origin
#List of the points being hit by this ray 
    x = np.array(o)
#direction cosines of the ray
    ll = random.choice(penetration_len) #select a random penetration length from the data
    l=ll[0]
    
    #l = (1+np.random.random(1)[0])*D

    R_theta = np.random.random(1)[0]
    
    theta = pi*R_theta# pi to make sure that the ray toward up and dose not go down, 
    # the starting point is (0,0), thus there is no need for rotation matrix
    dir_cos = np.array([cos(theta),sin(theta)])
    Dir_cosines = np.array([0,0])
    Dir_cosines = np.vstack((Dir_cosines,dir_cos)) #array to save all the direction cosines of the ray
    ip = np.array([dir_cos[0]*l,dir_cos[1]*l]) # first hit point (end point of ray, the firist point is (0,0))
    x = np.vstack((x,o)) #first point in x is redundant and is added just to make cade easier
    ep = ip # ep is the end point of each step
    sphere_center=np.array([0,0]) 
    I = 1 #Intensity or energy of the ray
    ii = 1  
    iii= 0
    
    all_point=x[:]
  
    while I > 1E-10:
    

        
        
        
        overlap = False
# skipping the first step when end point is already known       
        if len(x)> 2:
            ll = random.choice(penetration_len)
            l=ll[0]
           #l = (0.5+np.random.random(1)[0])*D #in case we don't have data for penetration length
            ep = np.array([x[ii][0]+Dir_cosines[ii][0]*l,x[ii][1]+Dir_cosines[ii][1]*l]) #end point of the ray for this iteration
# end point it will be the firist point + l * direction
            
            
        if ep[1]< A[1] and ep[1]>0:# to make sure the last point in the domain of porous media
#random cone and circumferential angles to get a random point of the selected particle
            #randomly getting the center of the sphere being hit
            R_theta1 = np.random.random(1)[0]
            
            theta1 = pi*R_theta1# pi to make sure the sphere dose not intersect with ray twice

            
            
            

##coordinates of the center of the spheres  
            dir_cos_sec = np.array([cos(theta1),sin(theta1)])
            
# convert the direction of center of sphere to ground frame 
            
## Rotation matrix to get the direction in the ground frame
            
##            [rotation matrix][secondary frame direction]=[direction in ground frame]

##           [rotation matrix]=[cos (a) sin(a), [-sin(a)  cos(a)]] 
## a is the angle of rotation
## a is 90 - angle of vector (1) that the second vector rotated abour
## the vector (1) is the y+ direction of the second vector
## a should be releative to +x direction, so it should be from 0 to 360$
             
            last_vector=np.array([Dir_cosines[ii]])
            
            theta_x_axis=atan(abs(last_vector[0][1])/abs(last_vector[0][0]))
                
            if last_vector[0][1]>0 and last_vector[0][0]>0:
                theta_last_vector=theta_x_axis
            elif last_vector[0][1]>0 and last_vector[0][0]<0:
                theta_last_vector=pi-theta_x_axis
                    
            elif last_vector[0][1]<0 and last_vector[0][0]<0:
                theta_last_vector=pi+theta_x_axis
            else:
                theta_last_vector=2*pi-theta_x_axis
                    
           
            theta_of_vector=theta_last_vector
            angle_rotation=(pi/2)-theta_of_vector    
            rot=np.array([[cos(angle_rotation), sin(angle_rotation)],[-sin(angle_rotation), cos(angle_rotation)]])
            Dir_cos_grd=np.matmul(rot, dir_cos_sec)
            
            
            
            
            # to find the center of sphere in the ground  frame= final point of ray + direction of center * D/2
            sphere_center_grd = np.array([Dir_cos_grd[0]*D/2 + ep[0],Dir_cos_grd[1]*D/2 + ep[1]])     
            sphere_center_grd = np.reshape(sphere_center_grd, (1, 2))
            sphere_center = np.vstack((sphere_center,sphere_center_grd))# adding sphere being hit to the list to keep track of the spheres being added, later add this to the main list
    
#code for checking the overlap
            #Code for checking whether the new sphere is overlaping with an existing sphere 
            dis_list = []
            B = []
            for a in sphere_center[1:-1]:
                dis = (sphere_center_grd[0][0] - a[0])**2 + (sphere_center_grd[0][1] - a[1])**2       
                dis_list.append(dis)
            for a in dis_list:
                if a<(((D+gab)**2)):
                    B.append(1)
                else:
                    B.append(0)
            if sum(B)>0.5:
                overlap = True
            # Code to check whether the emitted ray is hitting any of the existing spheres
#            p11= np.array([ep[0],ep[1])
#            p22= np.array([ep[0]+Dir_cosines[ii][0]*l,ep[1]+Dir_cosines[ii][1]*l])
            

            p11= np.array([ep[0]-Dir_cosines[ii][0]*l,ep[1]-Dir_cosines[ii][1]*l])# firsit point
            p22= np.array([ep[0],ep[1]]) 
            
            p11 = np.reshape(p11,(1,2))
            p22 = np.reshape(p22,(1,2))
            
            for a in sphere_center[1:-1]:# -2 because I want to exculde the last sphere the sphere that I generate from this iteration
                a = np.array(a)
                y = np.divide(p22 - p11, np.linalg.norm(p22 - p11))                
#                n1_ = np.squeeze(np.asarray(p11-a))
#                n1 = np.array([n1_[0] - a[0],n1_[1] - a[1]])
                
                n1 = np.squeeze(np.asarray(p11-a))
                
                n2 = np.squeeze(np.asarray(y))
                n3 = np.squeeze(np.asarray(a-p22))
                s = np.dot(n1, n2)
                t = np.dot(n3, n2)
                h = np.maximum.reduce([s, t, 0])
                c = np.cross(a - p11, y)
                w = LA.norm(c)
                Dis_center_to_line = sqrt(w**2 + h**2)
                if Dis_center_to_line < D/2:
                    overlap = True
#If any of the above conditions is true then do the tracing again and so deleting the previous step      
            if overlap==True:
                iii = iii+1
                sphere_center = np.delete(sphere_center,ii,0)
                if iii > 100:
                    
                    x = np.delete(x,ii,0)
                    Dir_cosines = np.delete(Dir_cosines,ii,0)
                    sphere_center = np.delete(sphere_center,ii-1,0)
                    ii = ii-1
                    iii = 0
                    continue
                continue# if tje overlap go back to while loop and generate another cirlce being hit be the ray
            # if the spheres continue to overlap more than 100 tiems then go to for loop and emit a new ray
            
        
        
#code for checking whether to do reflection or refraction
            normal = np.array([ep[0]-sphere_center[ii][0],ep[1]-sphere_center[ii][1]])# point on the sphere - center to find the normal
            norm = LA.norm(normal)
            normal_dir_cos = np.array([normal[0]/norm, normal[1]/norm])# we convert the normal to direction
            normal_dir_cos=np.reshape(normal_dir_cos,(1,2))
            normal_dir_cos=np.squeeze(normal_dir_cos)
            
            
        
            phi = acos(np.dot(-normal_dir_cos,Dir_cosines[ii]))# we multiply the normal by - to have the sharp angle
            phi2 = asin(nta_air*sin(phi)/nta)
            
            rho_parallel = (nta*cos(phi)-nta_air*cos(phi2))/(nta*cos(phi)+nta_air*cos(phi2))
            rho_perpendicular = (nta_air*cos(phi2)-nta*cos(phi))/(nta_air*cos(phi2)+nta*cos(phi))

            trans_parallel = (2*sin(phi2)*cos(phi))/(sin(phi+phi2)*cos(phi-phi2))
            trans_perpendicular = (2*sin(phi2)*cos(phi))/(sin(phi+phi2))
        
            rho_avg = (rho_parallel**2 + rho_perpendicular**2)/2
            trans_avg = (trans_parallel**2 + trans_perpendicular**2)/2
            
            rand1 = np.random.random(1)[0]

            iii = 0
            if rand1< rho_avg: # Diffuse reflection

                 #getting the center of the sphere being hit
                R_theta2 = np.random.random(1)[0]
                theta2 =pi*R_theta2
                
                
                reflected_dir_cos1 = np.array([cos(theta2),sin(theta2)])
#                reflected_dir_cos = np.matmul(rot,reflected_dir_cos1)
                
                theta_x_axis=atan(abs(normal_dir_cos[1])/abs(normal_dir_cos[0]))
                
                if normal_dir_cos[1]>0 and normal_dir_cos[0]>0:
                    theta_of_normal=theta_x_axis
                elif normal_dir_cos[1]>0 and normal_dir_cos[0]<0:
                    theta_of_normal=pi-theta_x_axis
                elif normal_dir_cos[1]<0 and normal_dir_cos[0]<0:
                    theta_of_normal=pi+theta_x_axis
                else:
                    theta_of_normal=2*pi-theta_x_axis
                    
                
                
                
                theta_of_vector=theta_of_normal
                angle_rotation=(pi/2)-theta_of_vector    
                rot=np.array([[cos(angle_rotation), sin(angle_rotation)],[-sin(angle_rotation), cos(angle_rotation)]])
                reflected_dir_cos=np.matmul(rot, reflected_dir_cos1)
            
            
            
                
                
#                reflected_dir_cos=np.array([cos(-(theta2-theta1+270*(pi/180))),sin(-(theta2-theta1+90*(pi/180)))])
                
                
                reflected_dir_cos = np.reshape(reflected_dir_cos, (1, 2))
                Dir_cosines = np.vstack((Dir_cosines,reflected_dir_cos))
                x = np.vstack((x,ep))
                ii = ii+1
                I = I - I*eta # loss of intensity due to collision
                all_point=np.vstack((all_point,ep))

            
            else: #refraction
                n = nta_air/nta
                cphi = np.dot(-normal_dir_cos,Dir_cosines[ii])
                phi = acos(np.dot(-normal_dir_cos,Dir_cosines[ii]))
                phi2 = asin(n*sin(phi))

                c1 = np.dot(-normal_dir_cos,Dir_cosines[ii])
                c2 = 1 - ((n**2)*(1-cphi**2))
                # refracted direction cosines
                refracted_dir_cos1 = np.array([(n)*(Dir_cosines[ii][0]) +(n*cphi-sqrt(c2))*normal_dir_cos[0], (n)*(Dir_cosines[ii][1]) +(n*cphi-sqrt(c2))*normal_dir_cos[1]])
                norm1 = LA.norm(refracted_dir_cos1)
                refracted_dir_cos = np.array([refracted_dir_cos1[0]/norm1,refracted_dir_cos1[1]/norm1])
                
                # find the final point after the ray has refracted and travelled through the sphere
                p1 = ep
                p2 = np.array([p1[0]+refracted_dir_cos[0]*1,p1[1]+refracted_dir_cos[1]*1])
                
                
                u = (p2[0] - p1[0])**2+(p2[1] - p1[1])**2
                v = -2*((p2[0] - p1[0])*(sphere_center[ii][0]-ep[0])) + -2*((p2[1] - p1[1])*(sphere_center[ii][1]-ep[1]))
                w = (sphere_center[ii][0]-ep[0])**2+(sphere_center[ii][1]-ep[1])**2-(D/2)**2
                t1 = (-v + sqrt(v**2-(4*u*w)))/(2*u)
                # t1= -v+  + becasue I want to find the farthest point where the ray exit the sphere
                ep2 = np.array([ep[0]+ t1*(p2[0] - p1[0]), ep[1]+ t1*(p2[1] - p1[1])])
                s = sqrt((ep2[0]-ep[0])**2 + (ep2[1]-ep[1])**2)
                normal2 = np.array([ep2[0]-sphere_center[ii][0],ep2[1]-sphere_center[ii][1]])
                norm2 = LA.norm(normal2)
                normal_dir_cos2 = np.array([normal2[0]/norm2, normal2[1]/norm2])
                b1 = nta/nta_air
                phi3 = acos(np.dot(normal_dir_cos2,refracted_dir_cos))
                phi4 = asin(b1*sin(phi3))
#                print(phi4)                         
                c3 = np.dot(normal_dir_cos2,refracted_dir_cos)
                c4 = 1 - ((b1**2)*(1-c3**2))
                refracted_dir_cos2 = np.array([(b1)*(refracted_dir_cos[0]) - (b1*c3-sqrt(c4))*normal_dir_cos2[0], (b1)*(refracted_dir_cos[1]) - (b1*c3-sqrt(c4))*normal_dir_cos2[1]])
                norm3 = LA.norm(refracted_dir_cos2)
                refracted_dir_cos3 = np.array([refracted_dir_cos2[0]/norm3,refracted_dir_cos2[1]/norm3])
#                print(acos(np.dot(normal_dir_cos2,refracted_dir_cos3)))
                #adding the final values to the list
                x = np.vstack((x,ep2))
                Dir_cosines = np.vstack((Dir_cosines,refracted_dir_cos3))
                ii = ii+1
                I = I - I*eta
                all_point1=np.vstack((all_point,ep))
                all_point=np.vstack((all_point1,ep2))


        elif ep[1]< 0:
            I = I
            E2.append(I)
            all_point=np.vstack((all_point,ep))
            #print("end2")
            break
        else:
            E1.append(I)
            all_point=np.vstack((all_point,ep))
            #print("end1")
            break
        
print ('Transmitted', sum(E1))
print ('reflected',sum(E2))

#print ('fail', fail)
#print ('sucess', sucess)

stop = timeit.default_timer()

print('Time: ', stop - start)


with open('all_point.csv', "w", newline = '') as output:  
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(all_point)
with open('sphere_center.csv', "w", newline = '') as output:  
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(sphere_center)