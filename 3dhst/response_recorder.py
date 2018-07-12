# Response recorder
# (Will Bowman, May 2017)
#
# record responses of visual inspection to a file
# e.g., classifying grism detections
#############################################################


import numpy as np
import matplotlib.pyplot as pl
import scipy as sp
#from scipy.integrate import quad
#import scipy.stats as spst
#import scipy.optimize as spopt
import math
from math import log10
from math import exp
from math import pi
#import statsmodels.api as sm
#import time

#from astropy.io import fits





def set_cols():
  'get number of columns, width of each column'

  while True:
    ncols = int(input("Enter the number of columns in output file: "))
    width = input("Enter width of each column (separated by whitespace): ")

    width = width.split()

    col_width = []
    for x in width:
      col_width.append(int(x))

    if ncols == len(col_width):
      break

  return (ncols, col_width)

def set_header(outfile, appending=False):
  'set and print header, if new file. otherwise, record existing header'
  if appending==True:
    with open(outfile,'r') as f:
      header = f.readline()[:-1]
  else:
    header = input("Enter file header (column names):\n")
    outfile.write('#  '+header+'\n#\n')

  return header


def get_response(outfile, ncols, col_width, header=''):
  '''outfile: opened file to record responses
  ncols: int, number of columns to record
  col_width: list of ints, len(col_width) = ncols, width of each column
  header: str, if not empty, print values'''

  if header != '':
    resp1 = input('Enter response ('+header+'):  ')
  else:
    resp1 = input('')

  while resp1 == '':
    resp1 = input("Empty response. Enter values again.\n")
  resp1 = resp1.split()
    

  if resp1[0].lower() == 'done':
    return 'done'

  while len(resp1) != ncols:
    resp1 = input("Wrong number of responses. Enter values again.\n")
    while resp1 == '':
      resp1 = input("Empty response. Enter values again.\n")
    resp1 = resp1.split()



  return resp1

def write_response(resp1, outfile, col_width):
  write_form = ''
  for val in col_width:
    write_form += ' %'+str(val)+'s'

  outfile.write( (write_form % tuple(resp1))+'\n' )


### MAIN


outfile = input("Enter name of output file:\n")

try:
  with open(outfile,'r') as f:
    print('Appending to existing file.')

  with open(outfile,'a') as f:

    ncols, col_width = set_cols()
    header = set_header(outfile, appending=True)

    i = -1
    while True:
      i += 1

      if i%20 == 0:
        print("Type 'done' to end writing")

      if True:
        resp = get_response(f, ncols, col_width, header)
#      else:
#        resp = get_response(f, ncols, col_width)

      if resp == 'done':
        break
      else:
        write_response(resp, f, col_width)


except FileNotFoundError:
  print('Writing new file.')
  with open(outfile,'w') as f:

    ncols, col_width = set_cols()
    header = set_header(f, appending=False)

    i = -1
    while True:
      i += 1

      if i%20 == 0:
        print("Type 'done' to end writing")

      if True:
        resp = get_response(f, ncols, col_width, header)
#      else:
#        resp = get_response(f, ncols, col_width)

      if resp == 'done':
        break
      else:
        write_response(resp, f, col_width)

