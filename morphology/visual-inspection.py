'''
Script to visually inspect HST cutouts of 3D-HST galaxies

Written by Will Bowman and Laurel Weiss
May 2018
'''


import numpy as np
import os
import aplpy
import matplotlib
import matplotlib.pyplot as pl
from astropy.table import Table
from astropy.cosmology import LambdaCDM
import astropy.units as u

#matplotlib.interactive(True)


def get_objects_info(file_in):

  x = np.genfromtxt(file_in, dtype=None)

  names = list(x.dtype.names)
  names[0] = 'field'
  names[1] = 'obj_id'
  names[2] = 'ra'
  names[3] = 'dec'
  names[4] = 'z'
  names[5] = 're_UV'
  names[6] = 're_opt'

  new_names = names[0:7]
  print("returning numpy structured array with columns:")
  print(new_names)
  x.dtype.names = names

  return x


def get_circle_rad(datalist, inp):
  z = datalist['z']
  re_UV = datalist['re_UV']
  re_opt = datalist['re_opt']

  scalar_list = []

  for i in z:
    cosmo = LambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3,Ode0=.7)
    scale = cosmo.kpc_proper_per_arcmin(i).to('kpc/arcsec')
    scalar = float(scale / (1.0 * (u.kpc / u.arcsec)))
    scalar_list.append(scalar)

  new_scale = []
  for i in scalar_list:
    a = (1/3600)*(1/i)
    new_scale.append(a)

  UV_rad = []
  opt_rad = []

  if inp == 'UV':
    for i, j in zip(new_scale, re_UV):
      a = i*j
      UV_rad.append(a)
    return UV_rad

  if inp == 'opt':
    for i, j in zip(new_scale, re_opt):
      b = i*j
      opt_rad.append(b)
    return opt_rad




def copy_images_script(field, obj_list, dir0, dir1, script_name):
  '''
  creates bash script to copy image cutout files from dir0 to dir1
  must make script executable (chmod +x script_name)
  then execute( ./script_name )

  field: can be string (if only one field) or list (if multiple)
  obj_list: list or array of integers
  '''

  if (isinstance(field, str)) & (len(obj_list)>1):
    field = [field]*len(obj_list)


  if os.path.exists(script_name):
    print('script file already exists. aborting.')
  else:
    with open(script_name, 'w') as fout:
  
      for i, obj in enumerate(obj_list):
        obj=int(obj)
        fname = '%s_%s.fits' % (field[i].upper(), str(obj).zfill(5))
        cmd = 'cp  %s%s  %s.' % (dir0, fname, dir1)
        fout.write(cmd+'\n')

def get_size_ref_file():
  t = Table.read('example_info.dat', format='ascii')
  return t


reffile = get_size_ref_file()


def get_center(field, obj_id, filter0):
  'filter0 = f814w --OR-- f160w'
  coo = np.genfromtxt( '%s/%s_%s.coo' % (filter0, field.upper(), str(int(obj_id)).zfill(5)) )
  return coo
 

def get_size(field, obj_id, array=reffile):
  i = np.where( (field.upper() == reffile['field']) & \
                (int(obj_id)==reffile['obj_id']) )[0][0]
  uv, opt = [ reffile[i]['re_UV'], reffile[i]['re_opt'] ]
  return [uv, opt]


#def plot_two_images(field, obj_id, filter1='f814w', filter2='f160w', ref=reffile, circles=True):
def plot_two_images(field, obj_id, UV_radius=None, opt_radius=None, 
                    filter1='f814w', filter2='f160w', ref=reffile, circles=True):
  fname = '%s_%s.fits' % (field.upper(), str(int(obj_id)).zfill(5))
  img1 = '%s/%s' % (filter1, fname)
  img2 = '%s/%s' % (filter2, fname)
  objname = '%s %s' % (field.upper(), str(int(obj_id)).zfill(5))
  ruv, ropt = get_size(field, obj_id)

  fig = pl.figure(figsize=(15, 7))

  f1 = aplpy.FITSFigure(img1, figure=fig, subplot=[0.1,0.1,0.35,0.8])
  f1.set_tick_labels_font(size='x-small')
  f1.set_axis_labels_font(size='small')
  f1.set_title('%s\nf814w (r$_{e, UV} = %s$)' % (objname, ruv))
  f1.show_grayscale()

  r1,d1 = get_center(field, obj_id, filter1)
  r1,d1 = f1.pixel2world(r1,d1)
  rad = 0.0012 # ~4.5 arcsecond

  f1.recenter( r1,d1, rad )

  f2 = aplpy.FITSFigure(img2, figure=fig, subplot=[0.5,0.1,0.35,0.8])
  f2.set_tick_labels_font(size='x-small')
  f2.set_axis_labels_font(size='small')
  f2.set_title('f160w (r$_{e, opt} = %s$)' % ropt)
  f2.show_grayscale()

  r2,d2 = get_center(field, obj_id, filter2)
  r2,d2 = f2.pixel2world(r2,d2)

  f2.recenter( r2,d2, rad)

  f2.hide_yaxis_label()
  f2.hide_ytick_labels()

  if circles:
    # plot circles of radius = half light radius
    if UV_radius:
      f1.show_circles(r1,d1, UV_radius,  edgecolor='red', facecolor='none',zorder=1)
      f2.show_circles(r2,d2, opt_radius, edgecolor='red', facecolor='none',zorder=1)
    # plot circles of radius = 1 arcsec
    f1.show_circles(r1,d1, 1/3600, edgecolor='blue', facecolor='none',zorder=1)
    f2.show_circles(r2,d2, 1/3600, edgecolor='blue', facecolor='none',zorder=1)
    # TEMPORARY - to check centroiding, plot a "dot" at center
    f1.show_markers(r1,d1)
    f2.show_markers(r2,d2) 


  fig.canvas.draw()


def view_field_images(datalist, option):
  iden = datalist['obj_id']
  temp_field = datalist['field']
  temp_index = 0
  if option == 1:
    start_index = 0
    field = temp_field[start_index:]
    index = start_index
    for j in field:
      if j == b'AEGIS':
        obj_id = iden[index]
        UV_radius = UV_rad[index]
        opt_radius = opt_rad[index]
        plot_two_images('AEGIS', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
        index += 1
      if j == b'COSMOS':
        obj_id = iden[index]
        UV_radius = UV_rad[index]
        opt_radius = opt_rad[index]
        plot_two_images('COSMOS', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
        index += 1
      if j == b'GOODSN':
        obj_id = iden[index]
        UV_radius = UV_rad[index]
        opt_radius = opt_rad[index]
        plot_two_images('GOODSN', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
        index += 1
  if option == 2:
    for i in iden:
      if i == start_image:
        start_index = temp_index
        break
      temp_index += 1
    field = temp_field[start_index:]
    index = start_index
    for j in field:
      if j == b'AEGIS':
        obj_id = iden[index]
        UV_radius = UV_rad[index]
        opt_radius = opt_rad[index]
        plot_two_images('AEGIS', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
        index += 1
      if j == b'COSMOS':
        obj_id = iden[index]
        UV_radius = UV_rad[index]
        opt_radius = opt_rad[index]
        plot_two_images('COSMOS', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
        index += 1
      if j == b'GOODSN':
        obj_id = iden[index]
        UV_radius = UV_rad[index]
        opt_radius = opt_rad[index]
        plot_two_images('GOODSN', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
        index += 1
  if option == 3:
    list_field = []
    for i in input_object_indices:
      a = temp_field[i]
      list_field.append(a)
    for i, j in zip(input_object_indices, list_field):
      if j == b'AEGIS':
        obj_id = iden[i]
        UV_radius = UV_rad[i]
        opt_radius = opt_rad[i]
        plot_two_images('AEGIS', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
      if j == b'COSMOS':
        obj_id = iden[i]
        UV_radius = UV_rad[i]
        opt_radius = opt_rad[i]
        plot_two_images('COSMOS', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()
      if j == b'GOODSN':
        obj_id = iden[i]
        UV_radius = UV_rad[i]
        opt_radius = opt_rad[i]
        plot_two_images('GOODSN', obj_id, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()



def main():
  file_in = input('Enter file for viewing: ')
  datalist = get_objects_info(file_in)
  opt_rad = get_circle_rad(datalist, 'opt')
  UV_rad = get_circle_rad(datalist, 'UV')

  temp = input('Do you want to view a list of objects?(y/n): ')
  if temp == 'y':
    option = 3
    input_list = input('Enter list to view: ')
    object_list = []
    for i in input_list.split():
      a = int(i)
      object_list.append(a)
    obj_id = datalist['obj_id']
    temp_index = 0
    input_object_indices = []
    for i in obj_id:
      if i in object_list:
        input_object_indices.append(temp_index)
      temp_index += 1

  if temp == 'n':
    begin = input('Start at beginning? (y/n): ')
    if begin == 'y':
      option = 1
    if begin == 'n':
      start_image = int(input('Enter starting image: '))
      option = 2

  view_field_images(datalist, option)

