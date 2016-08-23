#!/usr/bin/python

from scipy import ndimage
import numpy as np
from osgeo import gdal
import datetime
from scipy.stats import gmean
import math
import argparse
import os.path
import time
import hdf_ftp

# Start time
start = time.time()

# Maximum latitude default, minimum and maximum
DEF_MAX_LAT = 65.525
MIN_MAX_LAT = -90
MAX_MAX_LAT = 90

# Minimum latitude default, minimum and maximum
DEF_MIN_LAT = 65
MIN_MIN_LAT = -90
MAX_MIN_LAT = 90

# Maximum longitude default, minimum and maximum
DEF_MAX_LON = -146
MIN_MAX_LON = -180
MAX_MAX_LON = 180

# Minimum longitude default, minimum and maximum
DEF_MIN_LON = -148
MIN_MIN_LON = -180
MAX_MIN_LON = 180

# Reduction factor default, minimum and maximum
DEF_RED_FAC = 1
MIN_RED_FAC = 0
MAX_RED_FAC = 10

# Minimum window size default, minimum and maximum
DEF_MIN_KER = 5
MIN_MIN_KER = 0
MAX_MIN_KER = 100

# Maximum window size default, minimum and maximum
DEF_MAX_KER = 21
MIN_MAX_KER = 0
MAX_MAX_KER = 100

# Window observations default, minimum and maximum
DEF_WIN_OBV = 8
MIN_WIN_OBV = 1
MAX_WIN_OBV = 20

# Valid fraction of valid neighbors default, minimum and maximum
DEF_VLD_FRC = 0.25
MIN_VLD_FRC = 0.1
MAX_VLD_FRC = 1

# Decimal places of output default, minimum and maximum
DEF_DEC_PLC = 2
MIN_DEC_PLC = 0
MAX_DEC_PLC = 5

# Argument parser, run with -h for more info
parser = argparse.ArgumentParser()

# Command line arguments
parser.add_argument(
  "-v", "--verbose", help="turn on verbose output", action="store_true")

parser.add_argument(
  "-maxLat", "--maximumLatitude",
  help="the maximum latitude default:" + str(DEF_MAX_LAT) + " min:" + str(MIN_MAX_LAT) + " max:" + str(MAX_MAX_LAT),
  default=DEF_MAX_LAT, type=float)

parser.add_argument(
  "-minLat", "--minimumLatitude",
  help="the minimum latitude default:" + str(DEF_MIN_LAT) + " min:" + str(MIN_MIN_LAT) + " max:" + str(MAX_MIN_LAT),
  default=DEF_MIN_LAT, type=float)

parser.add_argument(
  "-maxLon", "--maximumLongitude",
  help="the maximum longitude default:" + str(DEF_MAX_LON) + " min:" + str(MIN_MAX_LON) + " max:" + str(MAX_MAX_LON),
  default=DEF_MAX_LON, type=float)

parser.add_argument(
  "-minLon", "--minimumLongitude",
  help="the minimum longitude default:" + str(DEF_MIN_LON) + " min:" + str(MIN_MIN_LON) + " max:" + str(MAX_MIN_LON),
  default=DEF_MIN_LON, type=float)

parser.add_argument(
  "-rf", "--reductionFactor",
  help="the reduction factor default:" + str(DEF_RED_FAC) + " min:" + str(MIN_RED_FAC) + " max:" + str(MAX_RED_FAC),
  default=DEF_RED_FAC, type=float)

parser.add_argument(
  "-minK", "--minimumKernel",
  help="the minimum kernel size default:" + str(DEF_MIN_KER) + " min:" + str(MIN_MIN_KER) + " max:" + str(MAX_MIN_KER),
  default=DEF_MIN_KER, type=int)

parser.add_argument(
  "-maxK", "--maximumKernel",
  help="the maximum kernel size default:" + str(DEF_MAX_KER) + " min:" + str(MIN_MAX_KER) + " max:" + str(MAX_MAX_KER),
  default=DEF_MAX_KER, type=int)

parser.add_argument(
  "-winObv", "--windowObservations",
  help="the amount of window observations default:" + str(DEF_WIN_OBV) + " min:" + str(MIN_WIN_OBV) + " max:" + str(
    MAX_WIN_OBV),
  default=DEF_WIN_OBV, type=int)

parser.add_argument(
  "-vldFrc", "--validFraction",
  help="valid fraction of valid observations default:" + str(DEF_VLD_FRC) + " min:" + str(MIN_VLD_FRC) + " max:" + str(
    MAX_VLD_FRC),
  default=DEF_VLD_FRC, type=float)

parser.add_argument(
  "-dec", "--decimal",
  help="Set the decimal places in the output default:" + str(DEF_DEC_PLC) + " min:" + str(MIN_DEC_PLC) + " max:" + str(
    MAX_DEC_PLC),
  default=DEF_DEC_PLC, type=float)

parser.add_argument(
  "-dir", "--directory",
  help="Set the directory to load HDF files from default:/",
  default=".", type=str)

# FTP arguments - these must mimic hdf_ftp argparse except for -v
parser.add_argument("-order", help="the data order id", type=str, nargs='+')
parser.add_argument("-dl", "--downloadLimit", help="limit the amount of HDF file pairs to download", default=0,
                    type=int)

# Parse the command line arguments
args = parser.parse_args()

# We have an FTP order to invoke
if (args.order):
  [hdf_ftp.main(order, args.downloadLimit, args.verbose) for order in args.order]

# Argument validation
if args.minimumLatitude < MIN_MIN_LAT:
  args.minimumLatitude = MIN_MIN_LAT
  if args.verbose:
    print("Raising minimum latitude to lower bound", MIN_MIN_LAT)
elif args.minimumLatitude > MAX_MIN_LAT:
  args.minimumLatitude = MAX_MIN_LAT
  if args.verbose:
    print("Lowering minimum latitude to upper bound", MAX_MIN_LAT)

if args.maximumLatitude < MIN_MAX_LAT:
  args.maximumLatitude = MIN_MAX_LAT
  if args.verbose:
    print("Raising maximum latitude to lower bound", MIN_MAX_LAT)
elif args.maximumLatitude > MAX_MAX_LAT:
  args.maximumLatitude = MAX_MAX_LAT
  if args.verbose:
    print("Lowering maximum latitude to upper bound", MAX_MAX_LAT)

if args.minimumLongitude < MIN_MIN_LON:
  args.minimumLongitude = MIN_MIN_LON
  if args.verbose:
    print("Raising minimum longitude to lower bound", MIN_MIN_LON)
elif args.minimumLongitude > MAX_MIN_LON:
  args.minimumLongitude = MAX_MIN_LON
  if args.verbose:
    print("Lowering minimum longitude to upper bound", MAX_MIN_LON)

if args.maximumLongitude < MIN_MAX_LON:
  args.maximumLongitude = MIN_MAX_LON
  if args.verbose:
    print("Raising maximum longitude to lower bound", MIN_MAX_LON)
elif args.maximumLongitude > MAX_MAX_LON:
  args.maximumLongitude = MAX_MAX_LON
  if args.verbose:
    print("Lowering maximum longitude to upper bound", MAX_MAX_LON)

if args.reductionFactor < MIN_RED_FAC:
  args.reductionFactor = MIN_RED_FAC
  if args.verbose:
    print("Raising reduction factor to lower bound", MIN_RED_FAC)
elif args.reductionFactor > MAX_RED_FAC:
  args.reductionFactor = MAX_RED_FAC
  if args.verbose:
    print("Lowering reduction factor to upper bound", MAX_RED_FAC)

if args.minimumKernel < MIN_MIN_KER:
  args.minimumKernel = MIN_MIN_KER
  if args.verbose:
    print("Raising minimum kernel size to lower bound", MIN_MIN_KER)
if args.minimumKernel > MAX_MIN_KER:
  args.minimumKernel = MAX_MIN_KER
  if args.verbose:
    print("Lowering minimum kernel size to upper bound", MAX_MIN_KER)

if args.maximumKernel < MIN_MAX_KER:
  args.maximumKernel = MIN_MAX_KER
  if args.verbose:
    print("Raising maximum kernel size to lower bound", MIN_MAX_KER)
if args.maximumKernel > MAX_MAX_KER:
  args.maximumKernel = MAX_MAX_KER
  if args.verbose:
    print("Lowering maximum kernel size to upper bound", MAX_MAX_KER)

if args.windowObservations < MIN_WIN_OBV:
  args.windowObservations = MIN_WIN_OBV
  if args.verbose:
    print("Raising window observation count to lower bound", MIN_WIN_OBV)
elif args.windowObservations > MAX_WIN_OBV:
  args.windowObservations = MAX_WIN_OBV
  if args.verbose:
    print("Lowering window observation count to upper bound", MAX_WIN_OBV)

if args.validFraction < MIN_VLD_FRC:
  args.validFraction = MIN_VLD_FRC
  if args.verbose:
    print("Raising valid fraction of observations to lower bound", MIN_VLD_FRC)
elif args.validFraction > MAX_VLD_FRC:
  args.validFraction = MAX_VLD_FRC
  if args.verbose:
    print("Lowering valid fraction of observations to upper bound", MAX_VLD_FRC)

if args.decimal < MIN_DEC_PLC:
  args.decimal = MIN_DEC_PLC
  if args.verbose:
    print("Raising decimal output to lower bound", MIN_DEC_PLC)
elif args.decimal > MAX_DEC_PLC:
  args.decimal = MAX_DEC_PLC
  if args.verbose:
    print("Lowering decimal output to upper bound", MAX_DEC_PLC)

# Verbose output configured settings
if args.verbose:
  print("Minimum latitude set to", args.minimumLatitude)
  print("Maximum latitude set to", args.maximumLatitude)
  print("Minimum longitude set to", args.minimumLongitude)
  print("Maximum longitude set to", args.maximumLongitude)
  print("Reduction factor set to", args.reductionFactor)
  print("Minimum kernel size set to", args.minimumKernel)
  print("Maximum kernel size set to", args.maximumKernel)
  print("Window observation count set to", args.windowObservations)
  print("Valid fraction of observations set to", args.validFraction)
  print("Decimal output set to", args.decimal)
  print("HDF loading directory set to", args.directory)


#
# Finds the number of adjacent cloud pixels
#
def adjCloud(kernel):
  nghbors = kernel[range(0, 4) + range(5, 9)]
  cloudNghbors = kernel[np.where(nghbors == 1)]
  nCloudNghbr = len(cloudNghbors)
  return nCloudNghbr


#
# Finds the number of adjacent water pixels
#
def adjWater(kernel):
  nghbors = kernel[range(0, 4) + range(5, 9)]
  waterNghbors = kernel[np.where(nghbors == 1)]
  nWaterNghbr = len(waterNghbors)
  return nWaterNghbr


#
# Creates a mask for context tests (must ignore pixels immediately to the right and left of center)
#
def makeFootprint(kSize):
  fpZeroLine = (kSize - 1) / 2
  fpZeroColStart = fpZeroLine - 1
  fpZeroColEnd = fpZeroColStart + 3
  fp = np.ones((kSize, kSize), dtype='int_')
  fp[fpZeroLine, fpZeroColStart:fpZeroColEnd] = -5
  return fp


#
# Returns the number of valid (non-background fire, non-cloud, non-water) neighbors for context tests
#
def nValidFilt(kernel, kSize, minKsize, maxKsize):
  nghbrCnt = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    fpMask = makeFootprint(kSize)
    kernel[np.where(fpMask < 0)] = -5
    nghbrs = kernel[np.where(kernel > 0)]
    nghbrCnt = len(nghbrs)

  return nghbrCnt


#
# Returns the number of neighbors rejected as background fires
#
def nRejectBGfireFilt(kernel, kSize, minKsize, maxKsize):
  nRejectBGfire = -4
  kernel = kernel.reshape((kSize, kSize))
  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    nRejectBGfire = len(kernel[np.where(kernel == -3)])

  return nRejectBGfire


#
# Returns number of neighbors rejected as water
#
def nRejectWaterFilt(kernel, kSize, minKsize, maxKsize):
  nRejectWater = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    nRejectWater = len(kernel[np.where(kernel == -1)])

  return nRejectWater


#
# Returns the number of 'unmasked water' neighbors
#
def nUnmaskedWaterFilt(kernel, kSize, minKsize, maxKsize):
  nUnmaskedWater = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if ((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-3, 0))):
    nUnmaskedWater = len(kernel[np.where(kernel == -6)])

  return nUnmaskedWater


#
# Generic ramp function used to calculate detection confidence
#
def rampFn(band, rampMin, rampMax):
  conf = 0
  confVals = []
  for bandVal in band:
    if rampMin < bandVal < rampMax:
      conf = (bandVal - rampMin) / (rampMax - rampMin)
    if bandVal >= rampMax:
      conf = 1
    confVals.append(conf)
  return np.asarray(confVals)


#
# Runs filters on progressively larger kernel sizes and then combines the result from the smallest kSize
#
def runFilt(band, filtFunc, minKsize, maxKsize):
  filtBand = band
  kSize = minKsize
  bandFilts = {}

  while kSize <= maxKsize:
    filtName = 'bandFilt' + str(kSize)
    filtBand = ndimage.generic_filter(filtBand, filtFunc, size=kSize, extra_arguments=(kSize, minKsize, maxKsize))
    bandFilts[filtName] = filtBand
    kSize += 2

  bandFilt = bandFilts['bandFilt' + str(minKsize)]
  kSize = minKsize + 2

  while kSize <= maxKsize:
    bandFilt[np.where(bandFilt == -4)] = bandFilts['bandFilt' + str(kSize)][np.where(bandFilt == -4)]
    kSize += 2

  return bandFilt


#
# Calculates mean and mean absolute deviation (MAD) of neighbouring pixels in a given band
# Valid neighbouring pixels must match the waterMask state of the corresponding waterMask center pixel
# Is used when both mean and MAD is required
def meanMadFilt(waterMask, rawband, minKsize, maxKsize, footprintx, footprinty, ksizes, minNcount, minNfrac):
  sizex, sizey = np.shape(rawband)
  bSize = (maxKsize - 1) / 2
  padsizex = sizex + 2 * bSize
  padsizey = sizey + 2 * bSize

  # This is the window which will be processed
  band = np.pad(rawband, ((bSize, bSize), (bSize, bSize)), mode='symmetric')

  # Get the window for the water mask as well
  waterBand = np.pad(waterMask, ((bSize, bSize), (bSize, bSize)), mode='symmetric')

  meanFilt = np.full([padsizex, padsizey], -4.0, dtype=np.float32)
  madFilt = np.full([padsizex, padsizey], -4.0, dtype=np.float32)

  divTable = 1.0 / np.arange(1, maxKsize * maxKsize, dtype=np.float64)
  divTable = np.insert(divTable, 0, 0)

  nmin = min(minNcount, minNfrac * minKsize * minKsize)

  for y in range(bSize, sizey + bSize):
    for x in range(bSize, sizex + bSize):

      # This is the center pixel of the window
      centerVal = band[x, y]

      # Get the corresponding center pixel of the water mask, 0 = land, 1 = water
      centerWaterVal = waterBand[x, y]

      if (centerVal not in range(-2, 0)):

        if meanFilt[x, y] == -4:

          # Get all possible neighbours of the current window
          neighbours = band[x + footprintx[0], y + footprinty[0]]

          # Remove neighbours which correspond to a value in the water mask which is not equal to the water mask centre value
          waterBandNeighbours = waterBand[x + footprintx[0], y + footprinty[0]]
          neighbours = neighbours[np.where(waterBandNeighbours == centerWaterVal)]

          neighbours = neighbours[np.where(neighbours > 0)]

          nn = len(neighbours)

          # The number of valid neighbours is more than what is required
          if (nn > nmin):
            bgMean = np.sum(neighbours) * divTable[nn]
            meanFilt[x, y] = bgMean
            meanDists = np.abs(neighbours - bgMean)
            bgMAD = np.sum(meanDists) * divTable[nn]
            madFilt[x, y] = bgMAD

  for i in range(1, len(ksizes)):

    nmin = min(minNcount, minNfrac * ksizes[i] * ksizes[i])

    for y in range(bSize, sizey + bSize):
      for x in range(bSize, sizex + bSize):

        # This is the center pixel of the window
        centerVal = band[x, y]

        # Get the corresponding center pixel of the water mask, 0 = land, 1 = water
        centerWaterVal = waterBand[x, y]

        if centerVal == -4:
          if meanFilt[x, y] == -4:

            # Get all possible neighbours of the current window
            neighbours = band[x + footprintx[0], y + footprinty[0]]

            # Remove neighbours which correspond to a value in the water mask which is not equal to the water mask centre value
            waterBandNeighbours = waterBand[x + footprintx[0], y + footprinty[0]]
            neighbours = neighbours[np.where(waterBandNeighbours == centerWaterVal)]

            neighbours = neighbours[np.where(neighbours > 0)]

            nn = len(neighbours)

            if (nn > nmin):
              bgMean = np.sum(neighbours) * divTable[nn]
              meanFilt[x, y] = bgMean
              meanDists = np.abs(neighbours - bgMean)
              bgMAD = np.sum(meanDists) * divTable[nn]
              madFilt[x, y] = bgMAD

  return meanFilt[bSize:-bSize, bSize:-bSize], madFilt[bSize:-bSize, bSize:-bSize]


#
# Main function for processing HDFs
#
def process(filMOD02, commandLineArgs, cwd):
  minNfrac = commandLineArgs.validFraction
  decimal = commandLineArgs.decimal
  minNcount = commandLineArgs.windowObservations
  maxKsize = commandLineArgs.maximumKernel
  minKsize = commandLineArgs.minimumKernel
  reductionFactor = commandLineArgs.reductionFactor
  maxLon = commandLineArgs.maximumLongitude
  minLon = commandLineArgs.minimumLongitude
  maxLat = commandLineArgs.maximumLatitude
  minLat = commandLineArgs.minimumLatitude

  # Value at which Band 22 saturates (L. Giglio, personal communication)
  b22saturationVal = 331
  increaseFactor = 1 + (1 - reductionFactor)
  waterFlag = -1
  cloudFlag = -2
  bgFlag = -3
  datsWdata = []

  # Coefficients for radiance calculations
  coeff1 = 119104200
  coeff2 = 14387.752
  lambda21and22 = 3.959
  lambda31 = 11.009
  lambda32 = 12.02

  # Layers for reading in HDF files
  layersMOD02 = ['EV_1KM_Emissive', 'EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB']
  layersMOD03 = ['Land/SeaMask', 'Latitude', 'Longitude', 'SolarAzimuth', 'SolarZenith', 'SensorAzimuth',
                 'SensorZenith']

  # meanMadFilt
  footprintx = []
  footprinty = []
  Ncount = []
  ksizes = []
  for s in range(minKsize, maxKsize + 2, 2):
    halfSize = (s - 1) / 2
    xlist = []
    ylist = []
    for x in range(-halfSize, halfSize + 1):
      for y in range(-halfSize, halfSize + 1):
        if x is 0:
          if abs(y) > 1:
            xlist.append(x)
            ylist.append(y)
        else:
          xlist.append(x)
          ylist.append(y)
    footprintx.append(np.array(xlist))
    footprinty.append(np.array(ylist))
    Ncount.append(len(xlist))
    ksizes.append(s)

  # Parse the HDF02
  filSplt = filMOD02.split('.')
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

  # Get the corresponding 03 HDF
  for filNamCandidate in HDF03:
    if datNam in filNamCandidate:
      filMOD03 = filNamCandidate
      break

  # Creates a blank dictionary to hold the full MODIS swaths
  fullArrays = {}

  for i, layer in enumerate(layersMOD02):

    file_template = 'HDF4_EOS:EOS_SWATH:%s:MODIS_SWATH_Type_L1B:%s'
    this_file = file_template % (filMOD02, layer)
    g = gdal.Open(this_file)
    if g is None:
      raise IOError
    metadataMOD02 = g.GetMetadata()
    dataMOD02 = g.ReadAsArray()

    if layer == 'EV_1KM_Emissive':
      B21index, B22index, B31index, B32index = 1, 2, 10, 11

      radScales = metadataMOD02["radiance_scales"].split(',')
      radScalesFlt = []
      for radScale in radScales:
        radScalesFlt.append(float(radScale))
      radScales = radScalesFlt
      del radScalesFlt

      radOffset = metadataMOD02["radiance_offsets"].split(',')
      radOffsetFlt = []
      for radOff in radOffset:
        radOffsetFlt.append(float(radOff))
      radOffset = radOffsetFlt
      del radOffsetFlt

      # Calculate temperature/reflectance based on scale and offset and correction term (L. Giglio, personal communication)
      B21, B22, B31, B32 = dataMOD02[B21index], dataMOD02[B22index], dataMOD02[B31index], dataMOD02[B32index]

      B21scale, B22scale, B31scale, B32scale = radScales[B21index], radScales[B22index], radScales[B31index], radScales[
        B32index]
      B21offset, B22offset, B31offset, B32offset = radOffset[B21index], radOffset[B22index], radOffset[B31index], \
                                                   radOffset[B32index]

      B21 = (B21 - B21offset) * B21scale
      T21 = coeff2 / (lambda21and22 * (np.log(coeff1 / (((math.pow(lambda21and22, 5)) * B21) + 1))))
      T21corr = 1.00009 * T21 - 0.05167
      fullArrays['BAND21'] = T21corr

      B22 = (B22 - B22offset) * B22scale
      T22 = coeff2 / (lambda21and22 * (np.log(coeff1 / (((math.pow(lambda21and22, 5)) * B22) + 1))))
      T22corr = 1.00010 * T22 - 0.05332
      fullArrays['BAND22'] = T22corr

      B31 = (B31 - B31offset) * B31scale
      T31 = coeff2 / (lambda31 * (np.log(coeff1 / (((math.pow(lambda31, 5)) * B31) + 1))))
      T31corr = 1.00046 * T31 - 0.09968
      fullArrays['BAND31'] = T31corr

      B32 = (B32 - B32offset) * B32scale
      T32 = coeff2 / (lambda32 * (np.log(coeff1 / (((math.pow(lambda32, 5)) * B32) + 1))))
      fullArrays['BAND32'] = T32

    if layer == 'EV_250_Aggr1km_RefSB':
      B1index, B2index = 0, 1

      refScales = metadataMOD02["reflectance_scales"].split(',')
      refScalesFlt = []
      for refScale in refScales:
        refScalesFlt.append(float(refScale))
      refScales = refScalesFlt
      del refScalesFlt

      refOffset = metadataMOD02["reflectance_offsets"].split(',')
      refOffsetFlt = []
      for refOff in refOffset:
        refOffsetFlt.append(float(refOff))
      refOffset = refOffsetFlt
      del refOffsetFlt

      B1, B2 = dataMOD02[B1index], dataMOD02[B2index]
      B1scale, B2scale = refScales[B1index], refScales[B2index]
      B1offset, B2offset = refOffset[B1index], refOffset[B2index]

      B1 = ((B1 - B1offset) * B1scale) * 1000
      B1 = B1.astype(int)
      B2 = ((B2 - B2offset) * B2scale) * 1000
      B2 = B2.astype(int)

      fullArrays['BAND1x1k'], fullArrays['BAND2x1k'] = B1, B2

    if layer == 'EV_500_Aggr1km_RefSB':
      B7index = 4

      refScales = metadataMOD02["reflectance_scales"].split(',')
      refScalesFlt = []
      for refScale in refScales:
        refScalesFlt.append(float(refScale))
      refScales = refScalesFlt
      del refScalesFlt

      refOffset = metadataMOD02["reflectance_offsets"].split(',')
      refOffsetFlt = []
      for refOff in refOffset:
        refOffsetFlt.append(float(refOff))
      refOffset = refOffsetFlt
      del refOffsetFlt

      B7 = dataMOD02[B7index]
      B7scale, B7offset = refScales[B7index], refOffset[B7index]
      B7 = ((B7 - B7offset) * B7scale) * 1000
      B7 = B7.astype(int)
      fullArrays['BAND7x1k'] = B7

  for i, layer in enumerate(layersMOD03):

    file_template = 'HDF4_EOS:EOS_SWATH:%s:MODIS_Swath_Type_GEO:%s'
    this_file = file_template % (filMOD03, layer)
    g = gdal.Open(this_file)
    if g is None:
      raise IOError
    if layer == 'Land/SeaMask':
      newLyrName = 'LANDMASK'
    elif layer == 'Latitude':
      newLyrName = 'LAT'
    elif layer == 'Longitude':
      newLyrName = 'LON'
    else:
      newLyrName = layer
    fullArrays[newLyrName] = g.ReadAsArray()

  # Clip area to bounding co-ordinates
  boundCrds = np.where((minLat < fullArrays['LAT']) & (fullArrays['LAT'] < maxLat) & (fullArrays['LON'] < maxLon) & (
    minLon < fullArrays['LON']))

  if np.size(boundCrds) > 0 and (np.min(boundCrds[0]) != np.max(boundCrds[0])) and (
        np.min(boundCrds[1]) != np.max(boundCrds[1])):

    boundCrds0 = boundCrds[0]
    boundCrds1 = boundCrds[1]
    min0 = np.min(boundCrds0)
    max0 = np.max(boundCrds0)
    min1 = np.min(boundCrds1)
    max1 = np.max(boundCrds1)

    # Creates a blank dictionary to hold the cropped MODIS data
    allArrays = {}  # Clipped to min/max lat/long
    for b in fullArrays.keys():
      cropB = fullArrays[b][min0:max0, min1:max1]
      allArrays[b] = cropB

    [nRows, nCols] = np.shape(allArrays['BAND22'])

    # Test for b22 saturation - replace with values from B21
    allArrays['BAND22'][np.where(allArrays['BAND22'] >= b22saturationVal)] = allArrays['BAND21'][
      np.where(allArrays['BAND22'] >= b22saturationVal)]

    # Day/Night flag (Giglio, 2016 Section 3.2)
    dayFlag = np.zeros((nRows, nCols), dtype=np.int)
    dayFlag[np.where(allArrays['SolarZenith'] < 8500)] = 1

    # Create water mask
    waterMask = np.zeros((nRows, nCols), dtype=np.int)
    waterMask[np.where(allArrays['LANDMASK'] != 1)] = waterFlag

    # Create cloud mask (Giglio, 2016 Section 3.2)
    cloudMask = np.zeros((nRows, nCols), dtype=np.int)
    cloudMask[((allArrays['BAND1x1k'] + allArrays['BAND2x1k']) > 1200)] = cloudFlag
    cloudMask[(allArrays['BAND32'] < 265)] = cloudFlag
    cloudMask[((allArrays['BAND1x1k'] + allArrays['BAND2x1k']) > 700) & (allArrays['BAND32'] < 285)] = cloudFlag

    cloudMask2 = np.zeros((nRows, nCols), dtype=np.int)
    cloudMask2[(allArrays['BAND2x1k'] > 250) & (allArrays['BAND32'] < 300)] = cloudFlag
    cloudMask2[np.where(waterMask == waterFlag)] = cloudFlag

    cloudMask[(cloudMask == cloudFlag) & (cloudMask2 == cloudFlag)] = cloudFlag

    # Mask clouds and water from input bands
    b21CloudWaterMasked = np.copy(allArrays['BAND21'])  # ONLY B21
    b21CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    b21CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    b22CloudWaterMasked = np.copy(allArrays['BAND22'])  # HAS B21 VALS WHERE B22 SATURATED
    b22CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    b22CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    b31CloudWaterMasked = np.copy(allArrays['BAND31'])
    b31CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    b31CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    deltaT = np.abs(allArrays['BAND22'] - allArrays['BAND31'])
    deltaTCloudWaterMasked = np.copy(deltaT)
    deltaTCloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    deltaTCloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    # After all the data has been read
    bgMask = np.zeros((nRows, nCols), dtype=np.int)

    ########################################
    # THIS IS BACKGROUND CHARACTERIZATION
    ########################################
    # Background fire test (Gilio 2003, Section 2.2.3, first paragraph)
    with np.errstate(invalid='ignore'):
      bgMask[np.where(
        (dayFlag == 1) & (allArrays['BAND22'] > (325 * reductionFactor)) & (deltaT > (20 * reductionFactor)))] = bgFlag
      bgMask[np.where(
        (dayFlag == 0) & (allArrays['BAND22'] > (310 * reductionFactor)) & (deltaT > (10 * reductionFactor)))] = bgFlag

    b21bgMask = np.copy(b21CloudWaterMasked)
    b21bgMask[np.where(bgMask == bgFlag)] = bgFlag

    b22bgMask = np.copy(b22CloudWaterMasked)
    b22bgMask[np.where(bgMask == bgFlag)] = bgFlag

    b31bgMask = np.copy(b31CloudWaterMasked)
    b31bgMask[np.where(bgMask == bgFlag)] = bgFlag

    deltaTbgMask = np.copy(deltaTCloudWaterMasked)
    deltaTbgMask[np.where(bgMask == bgFlag)] = bgFlag

    # Mean and mad filters - mad needed for confidence estimation
    b22meanFilt, b22MADfilt = meanMadFilt(waterMask, b22bgMask, maxKsize, minKsize, footprintx, footprinty, ksizes,
                                          minNcount, minNfrac)
    b22minusBG = np.copy(b22CloudWaterMasked) - np.copy(b22meanFilt)
    b31meanFilt, b31MADfilt = meanMadFilt(waterMask, b31bgMask, maxKsize, minKsize, footprintx, footprinty, ksizes,
                                          minNcount, minNfrac)
    deltaTmeanFilt, deltaTMADFilt = meanMadFilt(waterMask, deltaTbgMask, maxKsize, minKsize, footprintx, footprinty,
                                                ksizes, minNcount, minNfrac)
    b22bgRej = np.copy(allArrays['BAND22'])
    b22bgRej[np.where(bgMask != bgFlag)] = bgFlag
    b22rejMeanFilt, b22rejMADfilt = meanMadFilt(waterMask, b22bgRej, maxKsize, minKsize, footprintx, footprinty, ksizes,
                                                minNcount, minNfrac)

    # Potential fire test (Giglio 2016, Section 3.3)
    potFire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      potFire[(dayFlag == 1) & (allArrays['BAND22'] > (310 * reductionFactor)) & (deltaT > (10 * reductionFactor)) & (
        allArrays['BAND2x1k'] < (300 * increaseFactor))] = 1
      potFire[(dayFlag == 0) & (allArrays['BAND22'] > (305 * reductionFactor)) & (deltaT > (10 * reductionFactor))] = 1

    # CONTEXT TESTS - (Giglio 2016, Section 3.5)
    # The number associated with each test is the number of the equation in the paper

    # Absolute threshold test 1 [not contextual]
    absValTest = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      absValTest[(dayFlag == 1) & (allArrays['BAND22'] > (360 * reductionFactor))] = 1
      absValTest[(dayFlag == 0) & (allArrays['BAND22'] > (305 * reductionFactor))] = 1

    # Context fire test 2 (Giglio 2016, Section 3.5)
    deltaTMADfire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      deltaTMADfire[deltaT > (deltaTmeanFilt + (3.5 * deltaTMADFilt))] = 1

    # Context fire test 3 (Giglio 2016, Section 3.5)
    deltaTfire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      deltaTfire[np.where(deltaT > (deltaTmeanFilt + 6))] = 1

    # Context fire test 4 (Giglio 2016, Section 3.5)
    B22fire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      B22fire[(b22CloudWaterMasked > (b22meanFilt + (3 * b22MADfilt)))] = 1

    # Context fire test 5 (Giglio 2016, Section 3.5)
    B31fire = np.zeros((nRows, nCols), dtype=np.int)
    B31fire[(b31CloudWaterMasked > (b31meanFilt + b31MADfilt - 4))] = 1

    # Context fire test 6 (Giglio 2016, Section 3.5)
    B22rejFire = np.zeros((nRows, nCols), dtype=np.int)

    with np.errstate(invalid='ignore'):
      B22rejFire[(b22rejMADfilt > 5)] = 1

    # Combine tests to create tentative fires (Giglio 2016, section 3.5)
    fireLocTentative = deltaTMADfire * deltaTfire * B22fire

    fireLocB31andB22rejFire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      fireLocB31andB22rejFire[np.where((B22rejFire == 1) | (B31fire == 1))] = 1
    fireLocTentativeDay = potFire * fireLocTentative * fireLocB31andB22rejFire

    dayFires = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dayFires[(dayFlag == 1) & ((absValTest == 1) | (fireLocTentativeDay == 1))] = 1

    # Nighttime definite fire tests (Giglio 2016, Section 3.5)
    nightFires = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      nightFires[((dayFlag == 0) & ((fireLocTentative == 1) | absValTest == 1))] = 1

    # Sun glint rejection 7 (Giglio 2003, section 3.6.1)
    relAzimuth = allArrays['SensorAzimuth'] - allArrays['SolarAzimuth']
    cosThetaG = (np.cos(allArrays['SensorZenith']) * np.cos(allArrays['SolarZenith'])) - (
      np.sin(allArrays['SensorZenith']) * np.sin(allArrays['SolarZenith']) * np.cos(relAzimuth))
    thetaG = np.arccos(cosThetaG)
    thetaG = (thetaG / 3.141592) * 180

    # Sun glint test 7 (Giglio 2016, section 3.6.1)
    sgTest7 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest7[np.where(thetaG < 2)] = 1

    # Sun glint test 8 (Giglio 2016, section 3.6.1)
    sgTest8 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest8[np.where((thetaG < 8) & (allArrays['BAND1x1k'] > 100) & (allArrays['BAND2x1k'] > 200) & (
        allArrays['BAND7x1k'] > 120))] = 1

    # Sun glint test 9 (Giglio 2016, section 3.6.1)
    waterLoc = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      waterLoc[np.where(waterMask == waterFlag)] = 1
    nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size=3)
    nRejectedWater = runFilt(waterMask, nRejectWaterFilt, minKsize, maxKsize)
    with np.errstate(invalid='ignore'):
      nRejectedWater[np.where(nRejectedWater < 0)] = 0

    sgTest9 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest9[np.where((thetaG < 12) & ((nWaterAdj + nRejectedWater) > 0))] = 1

    sgAll = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgAll[(sgTest7 == 1) | (sgTest8 == 1) | (sgTest9 == 1)] = 1

    # Desert boundary rejection (Giglio 2003, section 2.2.7)
    nValid = runFilt(b22bgMask, nValidFilt, minKsize, maxKsize)
    nRejectedBG = runFilt(bgMask, nRejectBGfireFilt, minKsize, maxKsize)

    with np.errstate(invalid='ignore'):
      nRejectedBG[np.where(nRejectedBG < 0)] = 0

    # Desert boundary test 11 (Giglio 2003)
    dbTest11 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest11[np.where(nRejectedBG > (0.1 * nValid))] = 1

    # Desert boundary test 12 (Giglio 2003)
    dbTest12 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest12[(nRejectedBG >= 4)] = 1

    # Desert boundary test 13 (Giglio 2003)
    dbTest13 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest13[np.where(allArrays['BAND2x1k'] > 150)] = 1

    # Desert boundary test 14 (Giglio 2003)
    dbTest14 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest14[(b22rejMeanFilt < 345)] = 1

    # Desert boundary test 15 (Giglio 2003)
    dbTest15 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest15[(b22rejMADfilt < 3)] = 1

    # Desert boundary test 16 (Giglio 2003)
    dbTest16 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest16[(b22CloudWaterMasked < (b22rejMeanFilt + (6 * b22rejMADfilt)))] = 1

    # Reject anything that fulfills desert boundary criteria
    dbAll = dbTest11 * dbTest12 * dbTest13 * dbTest14 * dbTest15 * dbTest16

    # Coastal false alarm rejection (Giglio 2003, Section 2.2.8)
    with np.errstate(invalid='ignore'):
      ndvi = (allArrays['BAND2x1k'] + allArrays['BAND1x1k']) / (allArrays['BAND2x1k'] + allArrays['BAND1x1k'])
    unmaskedWater = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      unmaskedWater[((ndvi < 0) & (allArrays['BAND7x1k'] < 50) & (allArrays['BAND2x1k'] < 150))] = -6
      unmaskedWater[(bgMask == bgFlag)] = bgFlag
    Nuw = runFilt(unmaskedWater, nUnmaskedWaterFilt, minKsize, maxKsize)
    rejUnmaskedWater = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      rejUnmaskedWater[(absValTest == 0) & (Nuw > 0)] = 1

    # Combine all masks
    allFires = dayFires + nightFires  # All potential fires
    with np.errstate(invalid='ignore'):  # Reject sun glint, desert boundary, coastal false alarms
      allFires[(sgAll == 1) | (dbAll == 1) | (rejUnmaskedWater == 1)] = 0

    # If any fires have been detected, calculate Fire Radiative Power (FRP)
    if np.max(allFires) > 0:

      datsWdata.append(t)

      b22firesAllMask = allFires * allArrays['BAND22']
      b22bgAllMask = allFires * b22meanFilt

      b22maskEXP = b22firesAllMask.astype(float) ** 8
      b22bgEXP = b22bgAllMask.astype(float) ** 8

      frpMW = 4.34 * (10 ** (-19)) * (b22maskEXP - b22bgEXP)  # AREA TERM HERE

      frpMWabs = frpMW * potFire

      # Detection confidence (Giglio 2003, Section 2.3)
      cloudLoc = np.zeros((nRows, nCols), dtype=np.int)
      with np.errstate(invalid='ignore'):
        cloudLoc[np.where(cloudMask == cloudFlag)] = 1
      nCloudAdj = ndimage.generic_filter(cloudLoc, adjCloud, size=3)

      waterLoc = np.zeros((nRows, nCols), dtype=np.int)
      with np.errstate(invalid='ignore'):
        waterLoc[np.where(waterMask == waterFlag)] = 1
      nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size=3)

      # Fire detection confidence test 17
      z4 = b22minusBG / b22MADfilt

      # Fire detection confidence test 18
      zDeltaT = (deltaTbgMask - deltaTmeanFilt) / deltaTMADFilt

      with np.errstate(invalid='ignore'):
        firesNclouds = nCloudAdj[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        firesZ4 = z4[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        firesZdeltaT = zDeltaT[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        firesB22bgMask = b22bgMask[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        firesNwater = nWaterAdj[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        firesDayFlag = dayFlag[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]

      # Fire detection confidence test 19
      C1day = rampFn(firesB22bgMask, 310, 340)
      C1night = rampFn(firesB22bgMask, 305, 320)

      # Fire detection confidence test 20
      C2 = rampFn(firesZ4, 2.5, 6)

      # Fire detection confidence test 21
      C3 = rampFn(firesZdeltaT, 3, 6)

      # Fire detection confidence test 22 - not used for night fires
      C4 = 1 - rampFn(firesNclouds, 0, 6)  # zero adjacent clouds = zero confidence

      # Fire detection confidence test 23 - not used for night fires
      C5 = 1 - rampFn(firesNwater, 0, 6)

      confArrayDay = np.row_stack((C1day, C2, C3, C4, C5))
      detnConfDay = gmean(confArrayDay, axis=0)

      confArrayNight = np.row_stack((C1night, C2, C3))
      detnConfNight = gmean(confArrayNight, axis=0)

      detnConf = detnConfDay
      if 0 in firesDayFlag:
        detnConf[firesDayFlag == 0] = detnConfNight

      with np.errstate(invalid='ignore'):
        FRPx = np.where((allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900))[1]
        FRPsample = FRPx + min1
        FRPy = np.where((allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900))[0]
        FRPline = FRPy + min0
        FRPlats = allArrays['LAT'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPlons = allArrays['LON'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPT21 = allArrays['BAND22'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPT31 = allArrays['BAND31'][(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPMeanT21 = b22meanFilt[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPMeanT31 = b31meanFilt[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPMeanDT = deltaTmeanFilt[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPMADT21 = b22MADfilt[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRPMADT31 = b31MADfilt[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRP_MAD_DT = deltaTMADFilt[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRP_AdjCloud = nCloudAdj[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRP_AdjWater = nWaterAdj[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRP_NumValid = nValid[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]
        FRP_confidence = detnConf * 100
        FRPpower = frpMWabs[(allFires == 1) & (0 < frpMWabs) & (frpMWabs < 3900)]

      exportCSV = np.column_stack(
        [FRPline, FRPsample, FRPlats, FRPlons, FRPT21, FRPT31, FRPMeanT21, FRPMeanT31, FRPMeanDT, FRPMADT21, FRPMADT31,
         FRP_MAD_DT, FRPpower, FRP_AdjCloud, FRP_AdjWater, FRP_NumValid, FRP_confidence])
      hdr = '"FRPline","FRPsample","FRPlats","FRPlons","FRPT21","FRPT31","FRPMeanT21","FRPMeanT31","FRPMeanDT","FRPMADT21","FRPMADT31","FRP_MAD_DT","FRPpower","FRP_AdjCloud","FRP_AdjWater","FRP_NumValid","FRP_confidence"'
      os.chdir(cwd)
      np.savetxt(filMOD02.replace('hdf', '') + "csv", exportCSV, delimiter="\t", header=hdr,
                 fmt="%." + str(decimal) + "f")


# HDFs
cwd = os.getcwd()
os.chdir(args.directory)
HDF03 = [hdf for hdf in os.listdir('.') if ".hdf" in hdf and "D03" in hdf]
HDF02 = [hdf for hdf in os.listdir('.') if ".hdf" in hdf and "D02" in hdf]
[process(hdf02, args, cwd) for hdf02 in HDF02]

# End time
end = time.time()

if (args.verbose):
  print("Execution time " + str(end - start))
