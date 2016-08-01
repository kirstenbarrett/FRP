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

# Let's get a list of both the HDF03s and HDF02s
HDF03 = [hdf for hdf in order if ".hdf" in hdf and "D03" in hdf]
HDF02 = [hdf for hdf in order if ".hdf" in hdf and "D02" in hdf]

# Download all HDF02s with the corresponding HDF03s if they exist
for hdf02 in HDF02:

  # Parse the HDF02 in order to get the corresponding HDF03
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

  # Check to see if the HDF03 exists in the HDF03 list
  for filNamCandidate in HDF03:
    if datNam in filNamCandidate:
      hdf03 = filNamCandidate
      break

  # Both a HDF02 and HDF03 have been found
  if hdf03:
    if os.path.exists(hdf02) and order[order.index(hdf02) - 4] == os.path.getsize(hdf02):
      if args.verbose:
        print("Skipping download of " + hdf02)
    else:
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

    if os.path.exists(hdf03) and order[order.index(hdf03) - 4] == os.path.getsize(hdf03):
      if args.verbose:
        print("Skipping download of " + hdf03)
    else:
      if args.verbose:
        print("Attempting download of " + hdf03)
      fp = open(os.path.join('.', hdf03), "wb")
      curl = pycurl.Curl()
      curl.setopt(pycurl.URL, host + hdf03)
      curl.setopt(pycurl.WRITEDATA, fp)
      curl.perform()
      curl.close()
      fp.close()
      if args.verbose:
        print("Successfully downloaded " + hdf03)
  else:
    if args.verbose:
      print("Searching for a file pair to download")

  dlCount += 1

  if args.downloadLimit == dlCount:
    if args.verbose:
      print("HDF download limit reached")
    break

if args.verbose:
  print("FTP download of order successful")
