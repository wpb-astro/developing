# function to generate arrays of phot catalogs from 3dhst/skelton objects
# rows are objects, columns are filter

############################################################


from astropy.io import fits
from astropy.table import Table
import numpy as np
import pickle



def set_base_dir(base_in=None):
  if base_in:
    base_dir = base_in
  else:
    base_dir = '/home/will/research/3dhst/3dhst_grism_data/'
  return base_dir


def get_candidates(field, z='grism', pdf_width_constraint=True,base_dir=''):
  #, sorted_ids=False):
  '''return IDs of all objects with JH<26, 1.9<z<2.35
  if z==grism: use z_max_grism
        phot:  use z_peak_phot
  if width_constraint==True: require u68 - l68 < 0.05
    (only valid when using grism redshift)
  if sorted=True, sort ID's by continuum mag (bright to faint)'''

  base_dir = set_base_dir(base_dir)

  while z not in ['grism','phot']:
    z = input("invalid redshift param. enter 'grism' or 'phot'\n")

  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.zfit.linematched.fits'\
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data
  hdu.close()

  data = data[ data['jh_mag'] < 26 ]

  if z=='phot':
    z = 'z_peak_phot'
    data = data[ (data[z] < 2.35) & (data[z] > 1.9) ]
#    data = data[ (data[z] < 2.4) & (data[z] > 1.85) ]

  else:
    z = 'z_max_grism'
    z = 'z_best'
    data = data[ (data[z] < 2.35) & (data[z] > 1.9) ]
    if pdf_width_constraint:
      lo = 'z_grism_l68'
      hi = 'z_grism_u68'
#      lo = 'z_best_l68'
#      hi = 'z_best_u68'
      data = data[ (data[hi]-data[lo]) < 0.05 ]

  pid = data['phot_id']
  return pid


def get_detections(field, candidates=False, base_dir=''):
  '''return np array of ID, ra, dec, redshift of objects
     -OR- if candidates=True, return ids of initial selection (before defining sample!)'''

  base_dir = set_base_dir(base_dir)

  if not candidates:
    cat = np.genfromtxt(base_dir+'catalogs/'+field+'_positions.dat')
#    cat = cat[:,1:3]
    print("catalogs include objects with quality flags {Nline>1, cut<=1, contam<=1")
    print('columns {phot id, ra, dec, redshift}')
    return cat
  else:
    print('returning object IDs of all initially selected candidates')
    return np.genfromtxt( base_dir+'catalogs/'+field+'_candidates_ids.dat')


def get_detections2(field, qc_flag=2, cut_flag=1, contam_flag=1, base_dir='', id_only=True):
  '''qc_flag: minimum acceptable value
     cut_flag: max acceptable
     contam_flag: max acceptable

     return object IDs'''

  base_dir = set_base_dir(base_dir)

  qcat = np.genfromtxt(base_dir+field+'_3dhst_detection-qc.dat') 

  dv = 0.1

  cp = np.copy(qcat)
  cp = cp[ cp[:,1] > qc_flag - dv, :]
  cp = cp[ cp[:,2] < cut_flag + dv, :]
  cp = cp[ cp[:,3] < contam_flag + dv, :]

  if id_only:
    return cp[:,0]
  else:
    return cp
 
def get_detection_flags(field, base_dir=''):
  base_dir = set_base_dir(base_dir)
  qcat = np.genfromtxt(base_dir+field+'_3dhst_detection-qc.dat')
  return qcat

def remove_objects(catalog, col_pid, remove_list):
  '''given a catalog, remove objects with object ids in remove_list
     catalog: 2D numpy array
     col_pid: column of object ids in catalog array
              --OR-- is 1D array of object ids corresponding to catalog
     remove_list: 1D array of object ids to be removed


     return subset of catalog, with objects removed'''

  if type(col_pid) is int:
    if catalog.ndim < 2:
      try:
        rbool_keep = np.invert( \
                np.isin( catalog[ catalog.dtype.names[col_pid] ], remove_list) )
      except TypeError:
        print('Catalog has only one dimension? (doing nothing)')
        return False
    else:
      rbool_keep = np.invert( np.isin(catalog[:, col_pid], remove_list) )
    
  else:
    rbool_keep = np.invert( np.isin(col_pid, remove_list) )

  return catalog[rbool_keep]


def get_match_index(obj_id, obj_ra, pid, ra):
  '''given an object id, ra,
  return index of that object in a master array (pid, ra)'''

  index = np.where( (abs(obj_id - pid)<.5) & (abs(obj_ra - ra)<4) )[0]

  if len(index)==1:
    return index[0]
  else:
    print('ERROR')

def get_pids( catalog, col_pid ):
  '''HELPER FUNCTION -- reordering arrays'''
  if type(col_pid) is int:
    # consider case if array if structured array
    if catalog.ndim < 2:
      try:
        pid = catalog[ catalog.dtype.names[col_pid] ]
      except TypeError:
        print('Catalog has only one dimension? (doing nothing)')
        return False
    else:
      pid = catalog[:,col_pid]

  else:
    pid = col_pid

  return pid

def reorder_array(catalog, col_pid, ordered_ids):
  '''given catalog, reorder catalog according to ordered list of object ids
     catalog: 2D numpy array
     col_pid: index of column of object ids in catalog array
              --OR-- is 1D array of object ids corresponding to catalog
     ordered_ids: 1D array of object ids, in desired order

     return re-ordered catalog

     NOTE: objects in catalog but NOT in ordered_ids will be REMOVED'''

  cat_org = catalog.copy()

  if type(catalog)==Table:
    catalog=np.array(catalog)

  pid = get_pids(catalog, col_pid)
  obj_list = ordered_ids
  cat = catalog

  # set up output array: copy to retain type (structure vs. not)
  # cut length to match ordered_ids 
  sorted_cat = np.copy(cat[0:len(obj_list)])

  for i in range(len(obj_list)):
   try:
    if cat.ndim == 2:
      sorted_cat[i,:] = cat[ np.where( abs(obj_list[i] - pid)<.5 )[0][0], :]
    else:
      sorted_cat[i] = cat[ np.where( abs(obj_list[i] - pid)<.5 )[0][0]]
   except IndexError:
    if i < len(pid):
      print('object in catalog but NOT in sorted list --> excluding')

  if type(cat_org)==Table:
    sorted_cat = Table(sorted_cat)

  return sorted_cat


def reorder_two_arrays(cat1, colpid1, cat2, colpid2):
  '''given two arrays in different orders, containing different objects,
     return new arrays with ONLY objects that appear in both, 
     and with same order of objects'''

  pid1 = get_pids(cat1, colpid1)
  pid2 = get_pids(cat2, colpid2)

  if len(pid1) > len(pid2):
    rbool = np.isin( pid1, pid2)
    pid = pid1[ rbool ]
  else:
    rbool = np.isin( pid2, pid1 )
    pid = pid2[ rbool ]

  cat1 = reorder_array(cat1, colpid1, pid)
  cat2 = reorder_array(cat2, colpid2, pid)

  print('returning list of [catalog1, catalog2]')

  return [cat1, cat2]
 

def list_to_array(inlist, names=None):
  '''given list of 1D array, reform into 2D array'''

  out = np.column_stack( tuple(inlist) )

  if names:
    out = Table( out, names=names )

  return out


def reform_catalog(obj_list, measure_array, error_array):
  '''typically uses outputs from create_*_catalog() function,

     return array with columns {obj id, measure1, error1, measure2, error2, ....}'''


  nmeas = measure_array.shape[1]
  cat = np.zeros( (len(obj_list), 1 + 2*nmeas) )

  cat[:,0] = obj_list
  for i in range(nmeas):
    cat[:, 2*i + 1] = measure_array[:,i]
    cat[:, 2*i + 2] = error_array[:,i]

  return cat


def get_filter_dict(field, base_dir=''):
  ''' HELPER FUNCTION -- PHOTOMETRY CATALOG
  return dictionary of keys filter, values column of catalog'''

  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.cat.FITS' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )

  cols = hdu[1].columns.names
  hdu.close()

  filter_col_dict = {}
  for i in range( len(cols) ):
    if cols[i][0:2] == 'f_':
      filter_col_dict[cols[i]] = i

  return filter_col_dict


def get_filter_curve_dict(field, base_dir=''):
  ''' HELPER FUNCTION -- PHOTOMETRY CATALOG
  return dictionary of keys filter name in skelton, \
   values mcsed filter curve *.res name'''

  base_dir = set_base_dir(base_dir)

  with open(base_dir+'filter_dict.pickle', 'rb') as f:
    skelton_curves_dict = pickle.load(f)

  fc_dict = skelton_curves_dict[field]

  for x in list(fc_dict.keys()):
    fc_dict['f_'+str(x)] = fc_dict[x]
    del fc_dict[x]

  return fc_dict



def get_positions(field, obj_list, base_dir=''):
  '''input field name, list of phot IDs
  return RA,Dec  array'''
  
  base_dir = set_base_dir(base_dir)

  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.cat.FITS' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data
  hdu.close() 

  out_array = []
  for pid in obj_list:
    match_ind = np.where( abs(data['id'] - pid) < .1)[0][0]
    ra = data['ra'][match_ind]
    dec = data['dec'][match_ind]
    pos = [ra, dec]
    out_array.append( pos )

  out_array = np.array( out_array )

  return out_array


def get_redshifts(field, obj_list, base_dir=''):
  '''input field name, list of phot IDs
     return redshift  array'''

  base_dir = set_base_dir(base_dir)

  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.zfit.linematched.fits'\
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data
  hdu.close()

  rbool = np.isin(data['phot_id'], obj_list)
  data = data[rbool]

  z = reorder_array( data['z_max_grism'], data['phot_id'], obj_list )

  return z


def get_jhmags(field, obj_list, base_dir=''):
  '''input field name, list of phot IDs
     return redshift  array'''

  base_dir = set_base_dir(base_dir)

  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.zfit.linematched.fits'\
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data
  hdu.close()

  rbool = np.isin(data['phot_id'], obj_list)
  data = data[rbool]

  jh = reorder_array( data['jh_mag'], data['phot_id'], obj_list )

  return jh


def get_3dhst_masses(field, obj_list, base_dir='', remove_nan=True):
  '''input field name, list of phot IDs
     return log stellar mass array'''

  base_dir = set_base_dir(base_dir)

  cat = np.genfromtxt( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.zbest.fout'\
                    % (base_dir, field.upper(), field.lower(), field.lower()) ), \
                    dtype=None, names=True )

  rbool = np.isin(cat['id'], obj_list)
  data = cat[rbool]

  lmass = reorder_array( data['lmass'], data['id'], obj_list)

  if remove_nan:
    lm2 = lmass[ np.isnan(lmass)==False]
    if len(lm2)<len(lmass):
      print("CAUTION: removing nans --> object list does not correspond to array")
    return lm2
  else:
    return lmass


def get_UVJ_colors(field, obj_list, base_dir=''):
  base_dir = set_base_dir(base_dir)
  floc = ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.zbest.rf' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) )
  t = Table(np.genfromtxt(floc, dtype=None, names=True))

  t.rename_column('L153', 'Umag0')
  t.rename_column('L155', 'Vmag0')
  t.rename_column('L161', 'Jmag0')

  b = np.isin( t['id'], obj_list )

  n = t.colnames
  nindx = [0, 5, 9, 11]
  cols = [n[i] for i in nindx]

  tn = t[b][cols]

  n = tn.colnames
  for i, col in enumerate(n):
    if i == 0:
      continue
    else:
      tn[col] = 25. - 2.5 * np.log10( tn[col] )


  return tn



def get_skelton_cat(field, base_dir=''):
  base_dir = set_base_dir(base_dir)
  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.cat.FITS' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data

  hdu.close()
  return data


def get_3dhst_zfit_cat(field, base_dir=''):

  base_dir = set_base_dir(base_dir)
  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.zfit.linematched.fits' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data

  hdu.close()
  return data



def get_3dhst_linefit_cat(field, base_dir=''):

  base_dir = set_base_dir(base_dir)
  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.linefit.linematched.fits' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data

  hdu.close()
  return data



def create_phot_catalog(field, obj_list, filter_list='', base_dir=''):
  '''return array with rows of objects, columns of filter for SED fits
     parr_out: photometry array to be returned, columns phot bands, rows objects
     earr_out: errors corresponding to parr_out
     obj_list: list of photIDs
#     filter_names: *.res filter curve names

  field: {aegis, cosmos, goodsn, ...}
  obj_list: 1d array of photIDs for objects (int)
  filter_list: list of filters (str)'''

  base_dir = set_base_dir(base_dir)

  filter_dict = get_filter_dict(field, base_dir)
  if filter_list == '':
    filter_list = list( filter_dict.keys() )

#  filter_names = []
#
#  # dictionary of filter curve names
#  fc_dict = get_filter_curve_dict(field, base_dir)


  phot_file = ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.cat.FITS' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) )

  hdu = fits.open( phot_file )
  data = hdu[1].data
  hdu.close()

  rbool = np.isin( data.field(0), obj_list)
  pid = data['id'][rbool]

  # create output arrays
  parr_out = np.zeros( (len(obj_list), len(filter_list)+1) )
  earr_out = np.zeros( np.shape(parr_out) )

  parr_out[:,0] = pid
  earr_out[:,0] = pid

  for i,filt in enumerate(filter_list):
    # add photometry
    phot = data[filt][rbool]
    parr_out[:,i+1] = phot

    # add errors
    err = 'e'+filt[1:]
    phot_err = data[err][rbool]
    earr_out[:,i+1] = phot_err
    

  # want outputs in same order as input
  parr_out = reorder_array(parr_out, 0, obj_list)[:,1:]
  earr_out = reorder_array(earr_out, 0, obj_list)[:,1:]

  print("\nReturning list of [phot, phot_error, "+\
        "photID list, filter list] arrays")

  print('photometry in units of flux with zeropoint=25 AB, i.e.,\n'+\
        'mag_AB = 25 - 2.5 * log10(flux)')

  return [parr_out, earr_out, obj_list, filter_list]


def get_emislines():
  '''HELPER FUNCTION -- EMISSION LINE CATALOG'''
  return ['OII_FLUX', 'NeIII_FLUX', 'HeI_FLUX', 'Hd_FLUX', 'Hg_FLUX', \
          'OIIIx_FLUX', 'HeII_FLUX', 'Hb_FLUX', 'OIII_FLUX']


def create_emisline_catalog(field, obj_list, emis_line='', exclude_zero=False,
                            exclude_nonmeas=False, zero_val=False, base_dir=''):
  '''return flux line measurements for a given subset
  field: (aegis, cosmos, goodsn) (str)
  obj_list: list containing photIDs
  emis_line: list of emission lines to measure

  return np array of flux line measurements, units 1e-17 erg / s / cm2 '''

  base_dir = set_base_dir(base_dir)

  if emis_line == '':
    emis_line = get_emislines()

  hdu = fits.open( ('%s%s_WFC3_V4.1.5/%s_3dhst_v4.1.5_catalogs/%s_3dhst.v4.1.5.linefit.linematched.fits' \
                    % (base_dir, field.upper(), field.lower(), field.lower()) ) )
  data = hdu[1].data

  hdu.close()

  rbool = np.isin( data.field(0), obj_list)
  pid = data['number'][rbool]

  # create output arrays
  flux_out = np.zeros( (len(obj_list), len(emis_line)+1) )
  ferr_out = np.zeros( flux_out.shape )

  flux_out[:,0] = pid
  ferr_out[:,0] = pid

  for i,el in enumerate(emis_line):
    # line flux
    flux = data[el][rbool]
    flux_out[:,i+1] = flux

    # error
    err = el+'_ERR'
    ferr = data[err][rbool]
    ferr_out[:,i+1] = ferr 

  # want an array of line fluxes in same order as input obj_list
  flux_out = reorder_array(flux_out, 0, obj_list)[:,1:]
  ferr_out = reorder_array(ferr_out, 0, obj_list)[:,1:]

  # NOT RECOMMENDED, UNLESS USING ONLY ONE EMISSION LINE
  if exclude_nonmeas:
    obj_list_orig = np.copy(obj_list)
    print('exclude non measurements maybe be deprecated')
    missing=obj_list[ np.unique( np.where(flux_out<=0)[0]) ]

    flux_out = remove_objects( flux_out, obj_list_orig, missing)
    ferr_out = remove_objects( ferr_out, obj_list_orig, missing)
    obj_list = remove_objects( obj_list, obj_list_orig, missing)

#    for arr in [flux_out, ferr_out, obj_list]:
#      arr = remove_objects( arr, obj_list_orig, missing)
      
  if zero_val:
    print('set zero val deprecated')
#    cat[ cat<=0 ] = zero_val
  else:
    print('null value = -99')

  if exclude_zero:
    print('exclude zero deprecated')
#    obj_list = obj_list[ cat > zero_val]
#    cat = cat[cat > zero_val]

#  cat = cat*1e-17 # put in units of erg / s / cm2

  print('\nReturning list of [line flux, line error, '+\
        'obj ids, emission line names] arrays')

  print('line flux in units 1e-17 erg / s / cm2')

  return [flux_out, ferr_out, obj_list, emis_line]







#-----------------------------------------------------

def snr_cut(field, obj_list, min_snr=10, base_dir='', badlist=False):
  '''return object list including only objects with SNR>min_snr
  unless badlist=True --> then, return objects with SNR<min_snr'''

  base_dir = set_base_dir(base_dir)

  field_loc = {'AEGIS':215., 'COSMOS': 150., 'GOODSN': 189., 'GOODSS': 53.}

  pid = np.zeros( (len(obj_list),1) )
  pid[:,0] = obj_list

  phot = create_phot_catalog(field.lower(), pid, \
                            filter_list=['f_f140w','f_f160w'], base_dir=base_dir)
  snr = phot[0] / phot[1]

  # in absence of F140W measurements, use F160W (SNR agree between frames to within 10%)
  indices_without_f140w = phot[0][:,0] <0
  snr[indices_without_f140w,0] = snr[indices_without_f140w,1]
  snr = snr[:,0]

  good_list = pid[ snr >= min_snr ]
  bad_list = pid[snr < min_snr]
  if not badlist:
    return good_list
  else:
    return bad_list

#  rbool = np.isin( pid, good_list )
#  pid_out = pid[rbool[:,0]] #, :]

