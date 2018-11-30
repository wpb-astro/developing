'''
planning HETDEX pointings
adopted from code by Donghui Jeong

Author: Will Bowman
bowman@psu.edu

30Nov2018
'''


import numpy as np
import matplotlib.pyplot as pl
import astropy.units as u
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def angle2degrees(angle):
  '''
  convert angles from float or radians to degrees
  '''
  try:
    angle = angle.to('deg')
  except AttributeError:
    angle = angle * u.deg
  return angle

def Cos(value):
  '''value is in degrees'''
  value = angle2degrees(value)
  return np.cos( np.deg2rad( value ))

def Sin(value):
  '''value is in degrees'''
  value = angle2degrees(value)
  return np.sin( np.deg2rad( value ))

def Tan(value):
  '''value is in degrees'''
  value = angle2degrees(value)
  return np.tan( np.deg2rad( value ))


#def ArcCos(value):
#  '''return value in degrees'''
#  d


# a few lines at the beginning?

# basic HET numbers
HETlatitude = 30.681436 * u.degree
HETaltitude = 55. * u.degree 
HETangFreedom = 8.2 * u.degree

TDEN = HETlatitude - HETaltitude + 90. * u.degree

TDES = HETlatitude + HETaltitude - 90. * u.degree

TDENdegree = TDEN.copy()
TDESdegree = TDES.copy()

HETfoV = 11. * u.arcmin

HETP = Cos(HETlatitude) * Cos(HETaltitude)
HETQ = Sin(HETlatitude) * Sin(HETaltitude)


def FPangle(dec, track='E'):
  '''
  input declination (in degrees)
  return FP angle (currently in radians)
  '''

  # force input to have units of degrees
  dec = angle2degrees(dec)

  if (TDES <= dec <= TDEN):
    az = np.arccos( (Sin(dec)-HETQ)/HETP )
  elif (TDES - HETangFreedom <= dec <= TDES):
    az = (180. * u.deg).to('rad')
  elif (TDEN <= dec <= TDEN + HETangFreedom):
    az = 0. * u.rad
  else:
    print('Error')

  cosPA = ( Sin(HETaltitude) * Sin(dec) - Sin(HETlatitude) ) / (Cos(HETaltitude)*Cos(dec))

  # currently, returning position angle in units radians
  if (0. <= az.value <= np.pi):
    pa = np.arccos(cosPA)
  else:
    pa = 2*np.pi - np.arccos(cosPA)

  if track.upper()=='E':
    return pa
  elif track.upper()=='W':
    return 2.*np.pi*u.rad - pa
  else:
    print('Error: track keyword set incorrectly')

def Fgeodesic(ad, theta, phi):
  '''
  ad = (RA, Dec)
  inputs are in degrees, or radians, or float (if float, assume degrees)
  return RA, Dec (?)
  '''
  a0 = ad[0]
  d0 = ad[1]

  # force all angles in units degrees:
  degree_angles = []
  for value in [a0, d0, theta, phi]:
    degree_angles.append( angle2degrees(value) )
  a0, d0, theta, phi = degree_angles

  d = np.arcsin( Cos(theta)*Sin(d0) + Sin(theta)*Cos(d0)*Cos(phi) )
  a = a0 + np.arctan2( Sin(theta)*Sin(phi)/Cos(d), (Cos(theta) - Sin(d)*Sin(d0)) / (Cos(d)*Cos(d0)) )

  return [a.to('rad'), d]

# length of side of IFU (sIFU) and distance between IFU centers (dIFU)
sIFU = 50. * u.arcsec
dIFU = 100. * u.arcsec


# define IFU configuration
# wrt nominal IFU configuration, RHS = Tracker+Y, top = enclosure side A
fullIFUs = np.full((10,10),True)
IFU78 = fullIFUs.copy()

IFU78[0, 0:3] = False
IFU78[1,0] = False
IFU78[9,0:3] = False
IFU78[8,0] = False
IFU78[9,7:] = False
IFU78[8,9] = False
IFU78[0,7:] = False
IFU78[1,9] = False
IFU78[4:6, 3:6] = False

# 19-2 trimester
IFU58now = IFU78.copy()
IFU58now[4:8,0] = False
IFU58now[4:9,1] = False
IFU58now[4,2] = False
IFU58now[5,7] = False
IFU58now[8,7] = False
IFU58now[5:9,8] = False
IFU58now[4:8,9] = False

# 19-1 trimester
IFU50now = IFU58now.copy()
IFU50now[2:4, 8:] = False # 30, 31, 40, 41
IFU50now[2:4, 0] = False # 39, 49
IFU50now[3, 1] = False # 48
IFU50now[4, 8] = False


### Plotting IFUs on sphere
def IFUSpherePts(shotcentAD, ifucent, msize, pangle):
  '''final Polygon in units of degrees'''
  tanL = Tan( ifucent[0] )
  tanM = Tan( ifucent[1] ) 
  phi = np.arctan2( tanL, tanM )
  theta = np.arctan( tanM / Cos(phi) )
  adcent = Fgeodesic( shotcentAD, theta, phi + pangle)
  # add Polygon, Table...
  poly_table = [ Fgeodesic( adcent, msize/2.*np.sqrt(2.), pangle + (i*np.pi/2. +np.pi/4.)*u.rad) for i in range(1,5)]
  poly_table_degrees = [ [a.to('deg').value, d.to('deg').value] for a,d in poly_table ]
  coord_array = [c for d in poly_table_degrees for c in d]
#  return Polygon(poly_table_degrees, True) # need to add something like True as second argument?
  return coord_array

def IFUSphereCoords(whichIFU, shotcentAD, track='E'):

  pangle = FPangle( shotcentAD[1], track) 

  IFU_coord_list = []
  for nx in range(len(whichIFU)):
    for ny in range(len(whichIFU[nx])):
      if whichIFU[nx, ny]:
        ifucent = ( nx-4.5, ny-4.5 ) * dIFU
        IFUcoord = IFUSpherePts(shotcentAD, ifucent, sIFU, pangle)
        IFU_coord_list.append( IFUcoord )
  IFU_coord_list = np.array(IFU_coord_list)
  return IFU_coord_list 

   



#-------
#QQ: E/W track?

