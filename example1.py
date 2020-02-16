#!/usr/bin/python
# example of how to use ldbb.py

from ldbb import LDBB
from numpy import *
import matplotlib.pyplot as plt

# wavelength range of interest
lambda0 = linspace(200e-9,2000e-9,100)
epsD  = LDBB('Au','D',lambda0)
epsLD = LDBB('Au','LD',lambda0)
epsBB = LDBB('Au','BB',lambda0)

# scale wavelength for plot
lambda0 = lambda0*1e6

# LDBB returns permittivity, but we can easily convert it to refractive index

plt.subplot(2,2,1)
plt.title("Refractive Index of Au")
plt.ylabel('n')
plt.xlabel('wavelength (microns)')
plt.plot(lambda0,real(sqrt(epsD)), label="D")
plt.plot(lambda0,real(sqrt(epsBB)), label="LD")
plt.plot(lambda0,real(sqrt(epsLD)), label="BB")
plt.legend()

plt.subplot(2,2,2)
plt.ylabel('k')
plt.xlabel('wavelength (microns)')
plt.plot(lambda0,imag(sqrt(epsD)),label="D")
plt.plot(lambda0,imag(sqrt(epsBB)),label="LD")
plt.plot(lambda0,imag(sqrt(epsLD)),label="BB")
plt.legend()

plt.subplot(2,2,3)
nD=real(sqrt(epsD))
kD=imag(sqrt(epsD))
nBB=real(sqrt(epsBB))
kBB=imag(sqrt(epsBB))
nLD=real(sqrt(epsLD))
kLD=imag(sqrt(epsLD))



aD=((kD**2+(nD-1)**2)/(kD**2+(nD+1)**2))
aLD=((kLD**2+(nLD-1)**2)/(kLD**2+(nLD+1)**2))
aBB=((kBB**2+(nBB-1)**2)/(kBB**2+(nBB+1)**2))

plt.ylabel('absortvity')
plt.plot(lambda0,aD,label="D")
plt.plot(lambda0,aLD,label="LD")
plt.plot(lambda0,aBB,label="BB")

plt.xlabel('wavelength (microns)')

plt.show()









#### BEGIN EXAMPLE ###
##!/usr/bin/python
## example of how to use ldbb.py - text version
#
#from ldbb import LDBB
#from numpy import *
#
## wavelength range of interest (in meters) and number of points
#lambda0 = logspace(log10(0.2066e-6),log10(12.40e-6),200)
##epsD = LDBB('Au','D',lambda0)
##epsLD = LDBB('Au','LD',lambda0)
#epsBB = LDBB('Ti','BB',lambda0)
#
#
## LDBB returns permittivity, but we can easily convert it to refractive index
#for l,e in zip(lambda0,epsBB):
#    print(l*1e6, real(sqrt(e)), imag(sqrt(e)))
#
#### END EXAMPLE ###