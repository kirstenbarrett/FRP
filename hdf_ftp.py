#!/usr/bin/python

import argparse
import pycurl
import os.path
from io import BytesIO
import datetime

# Argument parser, run with -h for more info
parser = argparse.ArgumentParser()

# The order id argument
parser.add_argument("ORDER", help="the data order id", type=str)
# Max download count for HDFs
parser.add_argument("-dl", "--downloadLimit", help="limit the amount of HDF files to download", default=0, type=int)
# Verbosity output
parser.add_argument("-v", "--verbose", help="turn on verbose output", action="store_true")

args = parser.parse_args()

if args.verbose:
  print("Connecting to order " + args.ORDER)

# Build the ftp host with the order id
host = "ftp://ladsweb.nascom.nasa.gov/orders/" + args.ORDER + "/"

# Initiate curl
c = pycurl.Curl()
c.setopt(pycurl.URL, host)

# String output buffer for curl return
output = BytesIO()
c.setopt(pycurl.WRITEFUNCTION, output.write)

# Execute curl and get the order from the output buffer
c.perform()
order = output.getvalue().decode('UTF-8').split()

dlCount = 0

# Let's get both the HDF03s and HDF02s
HDF03 = [hdf for hdf in order if ".hdf" in hdf and "D03" in hdf]
HDF02 = [hdf for hdf in order if ".hdf" in hdf and "D02" in hdf]

# Download all HDF02s with the corresponding HDF03s if they exist
for hdf02 in HDF02:

  # The HDF already exists skip the download
  if os.path.exists(hdf02) and args.verbose:
    print("Skipping download of " + hdf02)
    continue

  if args.verbose:
    print("Attempting download of " + hdf02)

  fp = open(os.path.join('.', hdf02), "wb")
  curl = pycurl.Curl()
  curl.setopt(pycurl.URL, host + hdf02)
  curl.setopt(pycurl.WRITEDATA, fp)
  curl.perform()
  curl.close()
  fp.close()

  if args.verbose:
    print("Successfully downloaded " + hdf02)
    print("Searching for corresponding HDF03 file")

  # Let's get the corresponding HDF03
  filSplt = hdf02.split('.')
  datTim = filSplt[1].replace('A', '') + filSplt[2]
  t = datetime.datetime.strptime(datTim, "%Y%j%H%M")

  julianDay = str(t.timetuple().tm_yday)
  jZeros = 3 - len(julianDay)
  julianDay = '0' * jZeros + julianDay
  yr = str(t.year)
  hr = str(t.hour)
  hrZeros = 2 - len(hr)
  hr = '0' * hrZeros + hr
  mint = str(t.minute)
  mintZeros = 2 - len(mint)
  mint = '0' * mintZeros + mint
  datNam = yr + julianDay + '.' + hr + mint

  hdf03found = False
  # Get the corresponding HDF03
  for filNamCandidate in HDF03:
    if datNam in filNamCandidate:
      hdf03 = filNamCandidate
      hdf03found = True
      break

  if args.verbose:
    if hdf03found:
      print("Corresponding HDF03 file found")
      print("Attempting download of " + hdf03)
      fp = open(os.path.join('.', hdf03), "wb")
      curl = pycurl.Curl()
      curl.setopt(pycurl.URL, host + hdf03)
      curl.setopt(pycurl.WRITEDATA, fp)
      curl.perform()
      curl.close()
      fp.close()
      print("Successfully downloaded " + hdf03)
    else:
      print("No corresponding HDF03 file found")

  dlCount += 1

  if args.downloadLimit == dlCount:
    if args.verbose:
      print("HDF download limit reached")
    break

if args.verbose:
  print("FTP download of order successful")
