import numpy as np
import os
import aplpy
import matplotlib.pyplot as pl
from astropy.table import Table
from astropy.cosmology import LambdaCDM
import astropy.units as u

file_in = input('Enter file for viewing: ')

def sort_snr(reffile):
  iden = reffile['obj_id']
  snr = reffile['f160w_snr']
  indices = list(range(len(iden)))
  data = zip(snr, indices)
  n_data = sorted(data, key=lambda tup: tup[0])
  s_snr = []
  s_indices = []
  for i, j in n_data:
    s_snr.append(i)
    s_indices.append(j)
  return s_snr, s_indices

def get_size_ref_file(file_in):
  t = Table.read(file_in, format='ascii')
  return t

reffile = get_size_ref_file(file_in)
#reffile = Table.read('UV_opt_info.dat',format='ascii')

def get_center(field, obj_id, filter0):
  'filter0 = f814w --OR-- f160w'
  coo = np.genfromtxt( '%s/%s_%s.coo' % (filter0, field.upper(), str(int(obj_id)).zfill(5)) )
  return coo

def get_size(reffile, field, obj_id):
  temp_uv = reffile['hlr_f814w_kpc']
  temp_opt = reffile['hlr_f160w_kpc']
  temp_field = reffile['field']
  temp_id = reffile['obj_id']
  index = 0
  for i,j in zip(temp_field, temp_id):
    if i == field and j == obj_id:
      final_index = index
      break
    index +=1
  uv = temp_uv[final_index]
  opt = temp_opt[final_index]
  return [uv, opt]

def get_circle_rad(reffile, inp):
  z = reffile['z']
  re_UV = reffile['hlr_f814w_kpc']
  re_opt = reffile['hlr_f160w_kpc']

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

opt_rad = get_circle_rad(reffile, 'opt')
UV_rad = get_circle_rad(reffile, 'UV')

def plot_two_images(field, obj_id, snr, UV_radius=None, opt_radius=None, 
                    filter1='f814w', filter2='f160w', ref=reffile, circles=True):
  fname = '%s_%s.fits' % (field.upper(), str(int(obj_id)).zfill(5))
  img1 = '%s/%s' % (filter1, fname)
  img2 = '%s/%s' % (filter2, fname)
  objname = '%s %s' % (field.upper(), str(int(obj_id)).zfill(5))
  SNR = str(snr)
  ruv, ropt = get_size(reffile, field, obj_id)

  fig = pl.figure(figsize=(15, 7))

  f1 = aplpy.FITSFigure(img1, figure=fig, subplot=[0.1,0.1,0.35,0.8])
  f1.set_tick_labels_font(size='x-small')
  f1.set_axis_labels_font(size='small')
  f1.set_title('%s\nf814w (r$_{e, UV} = %s$)' % (objname, ruv))
  f1.show_grayscale(vmax = 0.03)

  r1,d1 = get_center(field, obj_id, filter1)
  r1,d1 = f1.pixel2world(r1,d1)
  rad = 0.0012 # ~4.5 arcsecond

  f1.recenter( r1,d1, rad )

  f2 = aplpy.FITSFigure(img2, figure=fig, subplot=[0.5,0.1,0.35,0.8])
  f2.set_tick_labels_font(size='x-small')
  f2.set_axis_labels_font(size='small')
  f2.set_title('f160w SNR = %s\nf160w (r$_{e, opt} = %s$)' % (SNR, ropt))
  f2.show_grayscale(vmax = 0.03)

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
 #   f1.show_markers(r1,d1)
 #   f2.show_markers(r2,d2) 

  fig.canvas.draw()

def view_field_images(reffile, option):
  iden = reffile['obj_id']
  temp_field = reffile['field']
  temp_snr = reffile['f160w_snr']
  temp_index = 0
  if option == 1:
    start_index = 0
    field = temp_field[start_index:]
    index = start_index
    for j in field:
      obj_id = iden[index]
      UV_radius = UV_rad[index]
      opt_radius = opt_rad[index]
      snr = temp_snr[index]
      plot_two_images(j, obj_id, snr, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
      pl.show()
      index += 1
  if option == 2:
    for i,j in zip(temp_field, iden):
      image = i+'_'+str(j)
      if image == start_image:
        start_index = temp_index
        break
      temp_index += 1
    field = temp_field[start_index:]
    index = start_index
    for j in field:
      obj_id = iden[index]
      UV_radius = UV_rad[index]
      opt_radius = opt_rad[index]
      snr = temp_snr[index]
      plot_two_images(j, obj_id, snr, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
      pl.show()
      index += 1
  if option == 3:
    list_field = []
    for i in input_object_indices:
      a = temp_field[i]
      list_field.append(a)
    for i in input_object_indices:
      field = temp_field[i]
      obj_id = iden[i]
      UV_radius = UV_rad[i]
      opt_radius = opt_rad[i]
      snr = temp_snr[i]
      plot_two_images(field, obj_id, snr, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
      pl.show()
  if option == 4:
    s_snr, s_indices = sort_snr(reffile)
    for i in s_indices:
      field = temp_field[i]
      obj_id = iden[i]
      UV_radius = UV_rad[i]
      opt_radius = opt_rad[i]
      snr = temp_snr[i]
      plot_two_images(field, obj_id, snr, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
      pl.show()
  if option == 5:
    s_snr, s_indices = sort_snr(reffile)
    for i, j in zip(s_indices, s_snr):
      if lower <= j <= upper:
        field = temp_field[i]
        obj_id = iden[i]
        UV_radius = UV_rad[i]
        opt_radius = opt_rad[i]
        snr = temp_snr[i]
        plot_two_images(field, obj_id, snr, UV_radius, opt_radius, filter1='f814w', filter2='f160w')
        pl.show()

temp = input('Do you want to view a list of objects?(y/n): ')
if temp == 'y':
  option = 3
  input_list = input('Enter list to view(field_id#): ')
  object_list = []
  field_list = []
  id_list = []
  for i in input_list.split():
    object_list.append(i)
  for i in object_list:
    a,b = i.split('_')
    field_list.append(a)
    id_list.append(int(b))
  obj_id = reffile['obj_id']
  field = reffile['field']
  temp_index = 0
  input_object_indices = []
  old_list = []
  new_list = []
  for i,j in zip(field, obj_id):
    old_list.append(i+str(j))
  for i,j in zip(field_list, id_list):
    new_list.append(i+str(j))
  for i in old_list:
    if i in new_list:
      input_object_indices.append(temp_index)
    temp_index += 1
  
if temp == 'n':
  temp2 = input('Sort by SNR?(y/n): ')
  if temp2 == 'y':
    temp3 = input('View a range?(y/n): ')
    if temp3 == 'n':
      option = 4
    if temp3 == 'y':
      range_list = []
      rang = input('Enter range(lower upper): ') #enter two numbers separated by a SPACE
      for i in rang.split():
        a = float(i)
        range_list.append(a)
      lower = range_list[0]
      upper = range_list[1]
      option = 5
  if temp2 == 'n':
    begin = input('Start at beginning? (y/n): ')
    if begin == 'y':
      option = 1
    if begin == 'n':
      start_image = int(input('Enter starting image: '))
      option = 2

view_field_images(reffile, option)

