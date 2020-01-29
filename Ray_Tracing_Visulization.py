# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 00:13:02 2020

@author: hssharadga
"""
import pygame
#pygame.init()

radius=50.0
W=2483
H=2483/2


image = pygame.display.set_mode((3000,1500))


for a in sphere_center:
    pygame.draw.circle(image, (255, 255, 255), (a[0], a[1]), radius, 1)
	
#pygame.image.save(image, "circle.png")
 