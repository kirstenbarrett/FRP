#!/usr/bin/python

import argparse
import pycurl
import os.path
from io import BytesIO

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

# Download all HDFs in the order info
for info in order:

  if ".hdf" not in info:
    continue

  # The HDF already exists skip the download
  if os.path.exists(info) and args.verbose:
    print("Skipping download of " + info)
    continue

  if args.verbose:
    print("Attempting download of " + info)

  fp = open(os.path.join('.', info), "wb")
  curl = pycurl.Curl()
  curl.setopt(pycurl.URL, host + info)
  curl.setopt(pycurl.WRITEDATA, fp)
  curl.perform()
  curl.close()
  fp.close()

  if args.verbose:
    print("Successfully downloaded " + info)

  dlCount += 1

  if args.downloadLimit == dlCount:
    if args.verbose:
      print("HDF download limit reached")
    break

if args.verbose:
  print("FTP download of order successful")
