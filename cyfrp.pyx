# cython: profile=True

from scipy import ndimage
import numpy as np
import os
from osgeo import gdal
import datetime
from scipy.stats import gmean
import math
cimport numpy as np

#
# Finds the number of adjacent cloud pixels
#
cdef adjCloud(kernel):

  nghbors = kernel[range(0, 4) + range(5, 9)]
  cloudNghbors = kernel[np.where(nghbors == 1)]
  cdef int nCloudNghbr = len(cloudNghbors)
  return nCloudNghbr

#
# Finds the number of adjacent water pixels
#
cdef adjWater(kernel):

  nghbors = kernel[range(0, 4) + range(5, 9)]
  waterNghbors = kernel[np.where(nghbors == 1)]
  cdef int nWaterNghbr = len(waterNghbors)
  return nWaterNghbr

#
# Creates a mask for context tests (must ignore pixels immediately to the right and left of center)
#
cdef makeFootprint(int kSize):

  cdef float fpZeroLine = (kSize - 1) / 2
  cdef float fpZeroColStart = fpZeroLine - 1
  cdef float fpZeroColEnd = fpZeroColStart + 3

  fp = np.ones((kSize, kSize), dtype='int_')
  fp[fpZeroLine, fpZeroColStart:fpZeroColEnd] = -5
  return fp

#
# Returns the number of valid (non-background fire, non-cloud, non-water) neighbors for context tests
#
cdef nValidFilt(kernel, int kSize, int minKsize, int maxKsize):

  cdef int nghbrCnt = -4
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
cdef nRejectBGfireFilt(kernel, int kSize, int minKsize, int maxKsize):

  cdef int nRejectBGfire = -4
  kernel = kernel.reshape((kSize, kSize))
  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    nRejectBGfire = len(kernel[np.where(kernel == -3)])

  return nRejectBGfire

#
# Returns number of neighbors rejected as water
#
cdef nRejectWaterFilt(kernel, int kSize, int minKsize, int maxKsize):

  cdef int nRejectWater = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    nRejectWater = len(kernel[np.where(kernel == -1)])

  return nRejectWater

#
# Return the number of neighbours rejected as coastline
#
cdef nRejectCoastFilt(kernel, kSize, minKsize, maxKsize):

  cdef int nRejecCoast = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    nRejecCoast = len(kernel[np.where(kernel == -3.5)])

  return nRejecCoast

#
# Returns the number of 'unmasked water' neighbors
#
cdef nUnmaskedWaterFilt(kernel, int kSize, int minKsize, int maxKsize):

  cdef int nUnmaskedWater = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if ((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-3, 0))):
    nUnmaskedWater = len(kernel[np.where(kernel == -6)])

  return nUnmaskedWater

#
# Generic ramp function used to calculate detection confidence
#
cdef rampFn(band, rampMin, rampMax):

  cdef float conf = 0
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
cdef runFilt(band, filtFunc, int minKsize, int maxKsize):

  filtBand = band
  cdef int kSize = minKsize
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
#
cdef meanMadFilt(np.ndarray[np.float64_t, ndim=2] waterMask, np.ndarray[np.float64_t, ndim=2] rawband, int minKsize, int maxKsize, minNcount, float minNfrac, footprintx, footprinty, ksizes):

    cdef int sizex, sizey, bSize, padsizex, padsizey, i, x, y, nmin, nn
    cdef float centerVal, bgMean, centerWaterVal
    cdef np.ndarray[np.float64_t, ndim=1] meanDists, neighbours, waterBandNeighbours
    cdef np.ndarray[np.float64_t, ndim=2] meanFilt,madFilt
    cdef np.ndarray[np.float64_t, ndim=2] band, waterBand
    cdef np.ndarray[np.float64_t, ndim=1] divTable

    sizex, sizey = np.shape(rawband)
    bSize = (maxKsize-1)/2
    padsizex = sizex+2*bSize
    padsizey = sizey+2*bSize
    band = np.pad(rawband,((bSize,bSize),(bSize,bSize)),mode='symmetric')
    waterBand = np.pad(waterMask, ((bSize, bSize), (bSize, bSize)), mode='symmetric')
    meanFilt = np.full([padsizex,padsizey], -4.0, dtype=np.float64)
    madFilt = np.full([padsizex,padsizey], -4.0, dtype=np.float64)

    divTable = 1.0/np.arange(1,maxKsize*maxKsize, dtype=np.float64)
    divTable = np.insert(divTable,0,0)

    nmin = min(minNcount, minNfrac*minKsize*minKsize)
    for y in range(bSize, sizey+bSize):
        for x in range(bSize, sizex+bSize):

            centerVal = band[x,y]
            centerWaterVal = waterBand[x, y]

            if centerVal not in range(-2,0):
              if meanFilt[x,y]==-4:

                neighbours = band[x+footprintx[0], y+footprinty[0]]

                waterBandNeighbours = waterBand[x + footprintx[0], y + footprinty[0]]
                neighbours = neighbours[np.where(waterBandNeighbours == centerWaterVal)]

                neighbours = neighbours[np.where(neighbours>0)]

                nn = len(neighbours)
                if (nn > nmin):
                    bgMean = np.sum(neighbours)*divTable[nn]
                    meanFilt[x,y] = bgMean
                    meanDists = np.abs(neighbours- bgMean)
                    bgMAD = np.sum(meanDists)*divTable[nn]
                    madFilt[x,y] = bgMAD

    for i in range(1.0, len(ksizes)):
        nmin = min(minNcount, minNfrac*ksizes[i]*ksizes[i])
        for y in range(bSize,sizey+bSize):
            for x in range(bSize,sizex+bSize):

                centerVal = band[x,y]
                centerWaterVal = waterBand[x, y]

                if centerVal == -4:
                  if meanFilt[x,y]==-4:

                    neighbours = band[x+footprintx[i], y+footprinty[i]]

                    waterBandNeighbours = waterBand[x + footprintx[0], y + footprinty[0]]
                    neighbours = neighbours[np.where(waterBandNeighbours == centerWaterVal)]

                    neighbours = neighbours[np.where(neighbours>0)]

                    nn = len(neighbours)
                    if (nn > nmin):
                        bgMean = np.sum(neighbours)*divTable[nn]
                        meanFilt[x,y] = bgMean
                        meanDists = np.abs(neighbours- bgMean)
                        bgMAD = np.sum(meanDists)*divTable[nn]
                        madFilt[x,y] = bgMAD

    return meanFilt[bSize:-bSize,bSize:-bSize], madFilt[bSize:-bSize,bSize:-bSize]

#
# Main function for processing HDFs
#
cdef process(filMOD02, HDF03, float minLat, float maxLat, float minLon, float maxLon,
             int reductionFactor, int minNcount, float minNfrac, int minKsize, int maxKsize, int decimal,
             str cwd, str hdfDir):

  cdef np.ndarray[np.float64_t, ndim=2] dayFlag,waterMask,cloudMask
  cdef np.ndarray[np.float64_t, ndim=2] b21CloudWaterMasked,b22CloudWaterMasked
  cdef np.ndarray[np.float64_t, ndim=2] b31CloudWaterMasked,deltaTCloudWaterMasked
  cdef np.ndarray[np.float64_t, ndim=2] bgMask,b21bgMask,b22bgMask,b31bgMask
  cdef np.ndarray[np.float64_t, ndim=2] b22meanFilt,b22MADfilt
  cdef np.ndarray[np.float64_t, ndim=2] b31meanFilt,b31MADfilt
  cdef np.ndarray[np.float64_t, ndim=2] deltaTmeanFilt, deltaTMADFilt
  cdef np.ndarray[np.float64_t, ndim=2] b22rejMeanFilt,b22rejMADfilt

  # Value at which Band 22 saturates (L. Giglio, personal communication)
  cdef float b22saturationVal = 331
  cdef float increaseFactor = 1 + (1 - reductionFactor)
  cdef float waterFlag = -1
  cdef float cloudFlag = -2
  cdef float bgFlag = -3
  cdef float coastFlag = -3.5

  # Coefficients for radiance calculations
  cdef int coeff1 = 119104200
  cdef float coeff2 = 14387.752
  cdef float lambda21and22 = 3.959
  cdef float lambda31 = 11.009
  cdef float lambda32 = 12.02

  # Layers for reading in HDF files
  layersMOD02 = ['EV_1KM_Emissive', 'EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB']
  layersMOD03 = ['Land/SeaMask', 'Latitude', 'Longitude', 'SolarAzimuth', 'SolarZenith', 'SensorAzimuth', 'SensorZenith']

  # meanMadFilt footprint
  footprintx = []
  footprinty = []
  Ncount = []
  ksizes = []
  for s in range(minKsize, maxKsize+2,2):
    halfSize = (s-1)/2
    xlist = []
    ylist = []
    for x in range(-halfSize,halfSize+1):
      for y in range(-halfSize,halfSize+1):
        if x is 0:
          if abs(y)>1:
            xlist.append(x)
            ylist.append(y)
        else:
          xlist.append(x)
          ylist.append(y)
    footprintx.append(np.array(xlist))
    footprinty.append(np.array(ylist))
    Ncount.append(len(xlist))
    ksizes.append(s)

  # Get the corresponding HDF03
  filSplt = filMOD02.split('.')
  cdef str datTim = filSplt[1].replace('A', '') + filSplt[2]
  t = datetime.datetime.strptime(datTim, "%Y%j%H%M")

  cdef str julianDay = str(t.timetuple().tm_yday)
  cdef int jZeros = 3 - len(julianDay)
  julianDay = '0' * jZeros + julianDay
  cdef str yr = str(t.year)
  cdef str hr = str(t.hour)
  cdef int hrZeros = 2 - len(hr)
  hr = '0' * hrZeros + hr
  cdef str mint = str(t.minute)
  cdef int mintZeros = 2 - len(mint)
  mint = '0' * mintZeros + mint
  cdef str datNam = yr + julianDay + '.' + hr + mint

  # Get the corresponding 03 HDF
  filMOD03 = None
  for filNamCandidate in HDF03:
    if datNam in filNamCandidate:
      filMOD03 = filNamCandidate
      break

  # The HDF03 does not exist - exit as we don't process a solitary HDF02
  if filMOD03 is None:
    return

  # Creates a blank dictionary to hold the full MODIS swaths
  fullArrays = {}

  # Invalid mask
  invalidMask = None;

  for i, layer in enumerate(layersMOD02):

    file_template = 'HDF4_EOS:EOS_SWATH:%s:MODIS_SWATH_Type_L1B:%s'
    this_file = file_template % (filMOD02, layer)
    g = gdal.Open(this_file)
    if g is None:
      return
    metadataMOD02 = g.GetMetadata()
    dataMOD02 = g.ReadAsArray()

    # Initialise the invalid mask if it is not already
    if invalidMask is None:
      invalidMask = np.zeros_like(dataMOD02[1])

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

      # Create the invalid mask from raw data values
      invalidMask[(B21 == 65534)] = 1
      invalidMask[(B22 == 65534)] = 1
      invalidMask[(B31 == 65534)] = 1
      invalidMask[(B32 == 65534)] = 1

      B21scale, B22scale, B31scale, B32scale = radScales[B21index], radScales[B22index], radScales[B31index], radScales[
        B32index]
      B21offset, B22offset, B31offset, B32offset = radOffset[B21index], radOffset[B22index], radOffset[B31index], \
                                                   radOffset[B32index]

      B21 = (B21 - B21offset) * B21scale
      T21 = coeff2 / (lambda21and22 * (np.log(coeff1 / (((math.pow(lambda21and22, 5)) * B21) + 1))))
      T21corr = 1.00009 * T21 - 0.05167
      fullArrays['BAND21'] = T21corr #temperature
      fullArrays['R21'] = B21 #radiance

      B22 = (B22 - B22offset) * B22scale
      T22 = coeff2 / (lambda21and22 * (np.log(coeff1 / (((math.pow(lambda21and22, 5)) * B22) + 1))))
      T22corr = 1.00010 * T22 - 0.05332
      fullArrays['BAND22'] = T22corr #temp
      fullArrays['R22'] = B22  #radiance

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

      # Create the invalid mask from raw data values
      invalidMask[(B1 == 65534)] = 1
      invalidMask[(B2 == 65534)] = 1

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

      # Create the invalid mask from raw data values
      invalidMask[(B7 == 65534)] = 1

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
    croppedArrays = {}  # Clipped to min/max lat/long
    for b in fullArrays.keys():
      cropB = fullArrays[b][min0:max0, min1:max1]
      croppedArrays[b] = cropB

    # Crop the invalid mask
    invalidMask = invalidMask[min0:max0, min1:max1]

    [nRows, nCols] = np.shape(croppedArrays['BAND22'])

    # Test for B22 saturation - replace with values from B21
    croppedArrays['BAND22'][np.where(croppedArrays['BAND22'] >= b22saturationVal)] = croppedArrays['BAND21'][
      np.where(croppedArrays['BAND22'] >= b22saturationVal)]

    # Test for R22 saturation - replace with values from R21
    croppedArrays['R22'][np.where(croppedArrays['BAND22'] >= b22saturationVal)] = croppedArrays['R21'][
      np.where(croppedArrays['BAND22'] >= b22saturationVal)]

    # Day/Night flag (Giglio, 2016 Section 3.2)
    dayFlag = np.zeros((nRows, nCols), dtype=np.float64)
    dayFlag[np.where(croppedArrays['SolarZenith'] < 8500)] = 1

    # Create water mask
    waterMask = np.zeros((nRows, nCols), dtype=np.float64)
    waterMask[np.where(croppedArrays['LANDMASK'] != 1)] = waterFlag

    # Create cloud mask (Giglio, 2016 Section 3.2)
    cloudMask = np.zeros((nRows, nCols), dtype=np.float64)
    cloudMask[((croppedArrays['BAND1x1k'] + croppedArrays['BAND2x1k']) > 1200)] = cloudFlag
    cloudMask[(croppedArrays['BAND32'] < 265)] = cloudFlag
    cloudMask[((croppedArrays['BAND1x1k'] + croppedArrays['BAND2x1k']) > 700) & (croppedArrays['BAND32'] < 285)] = cloudFlag

    cloudMask2 = np.zeros((nRows, nCols), dtype=np.int)
    cloudMask2[(croppedArrays['BAND2x1k'] > 250) & (croppedArrays['BAND32'] < 300)] = cloudFlag
    cloudMask2[np.where(waterMask == waterFlag)] = cloudFlag

    cloudMask[(cloudMask == cloudFlag) & (cloudMask2 == cloudFlag)] = cloudFlag

    # Mask clouds and water from input bands
    b21CloudWaterMasked = np.copy(croppedArrays['BAND21'])  # ONLY B21
    b21CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    b21CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    b22CloudWaterMasked = np.copy(croppedArrays['BAND22'])  # HAS B21 VALS WHERE B22 SATURATED
    b22CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    b22CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    b31CloudWaterMasked = np.copy(croppedArrays['BAND31'])
    b31CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    b31CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    deltaT = np.abs(croppedArrays['BAND22'] - croppedArrays['BAND31'])
    deltaTCloudWaterMasked = np.copy(deltaT)
    deltaTCloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
    deltaTCloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

    # Create coastal mask
    coastalMask = np.zeros((nRows, nCols), dtype=np.float64)
    coastalMask[croppedArrays['LANDMASK'] == 2] = coastFlag

    # Potential fire test (Giglio 2016, Section 3.3)
    potFire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      potFire[(dayFlag == 1) & (croppedArrays['BAND22'] > (310 * reductionFactor)) & (deltaT > (10 * reductionFactor)) & (
        croppedArrays['BAND2x1k'] < (300 * increaseFactor)) & (invalidMask == 0)] = 1
      potFire[(dayFlag == 0) & (croppedArrays['BAND22'] > (305 * reductionFactor)) & (deltaT > (10 * reductionFactor)) & (invalidMask == 0)] = 1

    # Absolute threshold test 1 [not contextual]
    test1 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      test1[(potFire == 1) & (dayFlag == 1) & (croppedArrays['BAND22'] > (360 * reductionFactor)) & (invalidMask == 0)] = 1
      test1[(potFire == 1) & (dayFlag == 0) & (croppedArrays['BAND22'] > (320 * reductionFactor)) & (invalidMask == 0)] = 1

    # Background fire test (Gilio 2003, Section 2.2.3, first paragraph)
    bgMask = np.zeros((nRows, nCols), dtype=np.float64)
    with np.errstate(invalid='ignore'):
      bgMask[
        (potFire == 1) & (dayFlag == 1) & (croppedArrays['BAND22'] > (325 * reductionFactor)) & (
        deltaT > (20 * reductionFactor)) & (invalidMask == 0)] = bgFlag
      bgMask[
        (potFire == 1) & (dayFlag == 0) & (croppedArrays['BAND22'] > (310 * reductionFactor)) & (
        deltaT > (10 * reductionFactor)) & (invalidMask == 0)] = bgFlag

    b22bgMask = np.copy(b22CloudWaterMasked)
    b22bgMask[(potFire == 1) & (bgMask == bgFlag) & (invalidMask == 0)] = bgFlag

    b31bgMask = np.copy(b31CloudWaterMasked)
    b31bgMask[(potFire == 1) & (bgMask == bgFlag) & (invalidMask == 0)] = bgFlag

    deltaTbgMask = np.copy(deltaTCloudWaterMasked)
    deltaTbgMask[(potFire == 1) & (bgMask == bgFlag) & (invalidMask == 0)] = bgFlag

    # Mean and mad filters - mad needed for confidence estimation
    b22meanFilt, b22MADfilt = meanMadFilt(waterMask, b22bgMask, minKsize, maxKsize, minNcount, minNfrac, footprintx, footprinty, ksizes)
    b22minusBG = np.copy(b22CloudWaterMasked) - np.copy(b22meanFilt)
    b31meanFilt, b31MADfilt = meanMadFilt(waterMask, b31bgMask, minKsize, maxKsize, minNcount, minNfrac, footprintx, footprinty, ksizes)
    deltaTmeanFilt, deltaTMADFilt = meanMadFilt(waterMask, deltaTbgMask, minKsize, maxKsize, minNcount, minNfrac, footprintx, footprinty, ksizes)

    b22bgRej = np.copy(croppedArrays['BAND22'])
    b22bgRej[(potFire == 1) & (bgMask != bgFlag)] = bgFlag
    b22rejMeanFilt, b22rejMADfilt = meanMadFilt(waterMask, b22bgRej, minKsize, maxKsize, minNcount, minNfrac, footprintx, footprinty, ksizes)

    # CONTEXT TESTS - (Giglio 2016, Section 3.5)
    # The number associated with each test is the number of the equation in the paper

    # Context fire test 2 (Giglio 2016, Section 3.5)
    test2 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      test2[(potFire == 1) & (deltaT > (deltaTmeanFilt + (3.5 * deltaTMADFilt))) & (invalidMask == 0)] = 1

    # Context fire test 3 (Giglio 2016, Section 3.5)
    test3 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      test3[(potFire == 1) & (deltaT > (deltaTmeanFilt + 6)) & (invalidMask == 0)] = 1

    # Context fire test 4 (Giglio 2016, Section 3.5)
    test4 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      test4[(potFire == 1) & (b22CloudWaterMasked > (b22meanFilt + (3 * b22MADfilt))) & (invalidMask == 0)] = 1

    # Context fire test 5 (Giglio 2016, Section 3.5)
    test5 = np.zeros((nRows, nCols), dtype=np.int)
    test5[(potFire == 1) & (b31CloudWaterMasked > (b31meanFilt + b31MADfilt - 4)) & (invalidMask == 0)] = 1

    # Context fire test 6 (Giglio 2016, Section 3.5)
    test6 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      test6[(potFire == 1) & (b22rejMADfilt > 5) & (invalidMask == 0)] = 1

    # Combine tests to create tentative fires (Giglio 2016, section 3.5)
    tests2and3and4 = test2 * test3 * test4

    test5or6 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      test5or6[(test5 == 1) | (test6 == 1)] = 1
    fireLocTentativeDay = potFire * tests2and3and4 * test5or6

    dayFires = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dayFires[(potFire == 1) & (dayFlag == 1) & ((test1 == 1) | (fireLocTentativeDay == 1)) & (invalidMask == 0)] = 1

    # Nighttime definite fire tests (Giglio 2016, Section 3.5)
    nightFires = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      nightFires[(potFire == 1) & ((dayFlag == 0) & ((tests2and3and4 == 1) | test1 == 1)) & (invalidMask == 0)] = 1

    # Sun glint rejection 7 (Giglio 2003, section 3.6.1)
    relAzimuth = croppedArrays['SensorAzimuth'] - croppedArrays['SolarAzimuth']
    cosThetaG = (np.cos(croppedArrays['SensorZenith']) * np.cos(croppedArrays['SolarZenith'])) - (
      np.sin(croppedArrays['SensorZenith']) * np.sin(croppedArrays['SolarZenith']) * np.cos(relAzimuth))
    thetaG = np.arccos(cosThetaG)
    thetaG = (thetaG / 3.141592) * 180

    sgTest7 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest7[(potFire == 1) & (thetaG < 2)] = 1

    # Sun glint test 8 (Giglio 2016, section 3.6.1)
    sgTest8 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest8[(potFire == 1) & ((thetaG < 10) & (croppedArrays['BAND1x1k'] > 120) & (croppedArrays['BAND2x1k'] > 200)) & (
        croppedArrays['BAND7x1k'] > 120) & (invalidMask == 0)] = 1

    # Sun glint test 9 (Giglio 2016, section 3.6.1)
    waterLoc = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      waterLoc[(potFire == 1) & (waterMask == waterFlag)] = 1
    nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size=3)
    nRejectedWater = runFilt(waterMask, nRejectWaterFilt, minKsize, maxKsize)
    with np.errstate(invalid='ignore'):
      nRejectedWater[(potFire == 1) & (nRejectedWater < 0) & (invalidMask == 0)] = 0

    sgTest9 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest9[(potFire == 1) & ((thetaG < 15) & ((nWaterAdj + nRejectedWater) > 0)) & (invalidMask == 0)] = 1

    sgAll = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgAll[(sgTest7 == 1) | (sgTest8 == 1) | (sgTest9 == 1)] = 1

    # Desert boundary rejection (Giglio 2003, section 2.2.7)
    nValid = runFilt(b22bgMask, nValidFilt, minKsize, maxKsize)
    nRejectedBG = runFilt(bgMask, nRejectBGfireFilt, minKsize, maxKsize)

    with np.errstate(invalid='ignore'):
      nRejectedBG[(potFire == 1) & (nRejectedBG < 0) & (invalidMask == 0)] = 0

    # Desert boundary test 11 (Giglio 2003, section 2.2.7)
    dbTest11 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest11[(potFire == 1) & ((nRejectedBG > (0.1 * nValid))) & (invalidMask == 0)] = 1

    # Desert boundary test 12 (Giglio 2003, section 2.2.7)
    dbTest12 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest12[(potFire == 1) & (nRejectedBG >= 4) & (invalidMask == 0)] = 1

    # Desert boundary test 13 (Giglio 2003, section 2.2.7)
    dbTest13 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest13[(potFire == 1) & (croppedArrays['BAND2x1k'] > 150) & (invalidMask == 0)] = 1

    # Desert boundary test 14 (Giglio 2003, section 2.2.7)
    dbTest14 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest14[(potFire == 1) & (b22rejMeanFilt < 345) & (invalidMask == 0)] = 1

    # Desert boundary test 15 (Giglio 2003, section 2.2.7)
    dbTest15 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest15[(potFire == 1) & (b22rejMADfilt < 3) & (invalidMask == 0)] = 1

    # Desert boundary test 16 (Giglio 2003, section 2.2.7)
    dbTest16 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest16[(potFire == 1) & (b22CloudWaterMasked < (b22rejMeanFilt + (6 * b22rejMADfilt))) & (invalidMask == 0)] = 1

    # Reject anything that fulfills desert boundary criteria
    dbAll = dbTest11 * dbTest12 * dbTest13 * dbTest14 * dbTest15 * dbTest16

    # Coastal false alarm rejection (Giglio 2003, Section 2.2.8)
    with np.errstate(invalid='ignore'):
      ndvi = (croppedArrays['BAND2x1k'] - croppedArrays['BAND1x1k']) / (croppedArrays['BAND2x1k'] + croppedArrays['BAND1x1k'])
    unmaskedWater = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      unmaskedWater[(potFire == 1) & ((ndvi < 0) & (croppedArrays['BAND7x1k'] < 50) & (croppedArrays['BAND2x1k'] < 150))] = -6
      unmaskedWater[(potFire == 1) & (bgMask == bgFlag)] = bgFlag
    Nuw = runFilt(unmaskedWater, nUnmaskedWaterFilt, minKsize, maxKsize)
    rejUnmaskedWater = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      rejUnmaskedWater[(potFire == 1) & ((test1 == 0) & (Nuw > 0)) & (invalidMask == 0)] = 1

    # Forest clearing rejection (Giglio 2016, 3.6.4)
    fcRej = np.zeros((nRows, nCols), dtype=np.int)
    fcRej[(dayFires == 1) & (waterMask != waterFlag) & (b31CloudWaterMasked > (b31meanFilt + (3.7 * b31MADfilt)))
          & (croppedArrays['BAND2x1k'] > 280) & (b22CloudWaterMasked < 325)
          & (invalidMask == 0)] = 1

    # Water pixel coastal rejection (Giglio 2016, 3.6.5)

    # Land pixels excluded from the background
    excludedLand = np.zeros((nRows, nCols), dtype=np.int)
    excludedLand[(waterMask != waterFlag)] = bgFlag
    # Get excluded land for the contextual window
    NL = runFilt(excludedLand, nRejectBGfireFilt, minKsize, maxKsize)

    # Allow all fires which lie over water where NL is less than 0
    NL[(potFire == 1) & (waterMask == waterFlag) & (NL < 0) & (invalidMask == 0)] = 0

    # Number of coast pixels (always excluded) within the contextual window
    Nc = runFilt(coastalMask, nRejectCoastFilt, minKsize, maxKsize)
    with np.errstate(invalid='ignore'):
      Nc[(potFire == 1) & (Nc < 0) & (invalidMask == 0)] = 0

    # Reject any tentative fire pixel detected over water for which NL + Nc > 0 and test (1) is not satisfied
    wpcRej = np.zeros((nRows, nCols), dtype=np.int)
    wpcRej[(potFire == 1) & (NL + Nc > 0) & (test1 == 0) & (invalidMask == 0)] = 1

    # Combine all masks
    allFires = dayFires + nightFires  # All potential fires
    with np.errstate(invalid='ignore'):  # Reject all from rejection testing output
      allFires[(sgAll == 1) | (dbAll == 1) | (rejUnmaskedWater == 1) | (fcRej == 1) | (wpcRej == 1)] = 0

    # If any fires have been detected, calculate Fire Radiative Power (FRP)
    if np.max(allFires) > 0:

      ##################
      ##AREA CALCULATION
      ##################
      ##S = (I-hp)/H
      ##
      ##where:
      ##
      ##I is the zero-based pixel index
      ##hp is 1/2 the total number of pixels (zero-based)
      ##    (for MODIS each scan is 1354 "1km" pixels, 1353 zero-based, so hp = 676.5)
      ##H is the sensor altitude divided by the pixel size
      ##    (for MODIS altitude is approximately 700km, so for "1km" pixels, H = 700/1)

      I = np.indices((nRows, nCols))[1]
      hp = 676.6
      H = 700

      S = (I - hp) / H

      ##Compute the zenith angle:
      Z = np.arcsin(1.111 * np.sin(S))

      ##Compute the Along-track pixel size:
      Pn = 1  # Pixel size in km at nadir
      Pt = Pn * 9 * np.sin(Z - S) / np.sin(S)

      ##Compute the Along-scan pixel size:
      Ps = Pt / np.cos(Z)

      areaKmSq = Pt * Ps

      # FRP equation 16 (Giglio 2016, Section 3.8)

      StBo = 5.6704 * (10 ** (-8)) #Stephan Boltzmann constant
      alpha = 3.0 * (10 ** (-9))
      tau4 = 1

      R22meanFilt, notNeeded = meanMadFilt(waterMask, croppedArrays['R22'], maxKsize, minKsize, footprintx, footprinty, ksizes, minNcount, minNfrac)

      # For all pixels not just fires
      FRP2016 = ((areaKmSq * StBo) / (alpha * tau4)) * (croppedArrays['R22'] - R22meanFilt)

      cloudLoc = np.zeros((nRows, nCols), dtype=np.int)
      with np.errstate(invalid='ignore'):
        cloudLoc[cloudMask == cloudFlag] = 1
      nCloudAdj = ndimage.generic_filter(cloudLoc, adjCloud, size=3)

      waterLoc = np.zeros((nRows, nCols), dtype=np.int)
      with np.errstate(invalid='ignore'):
        waterLoc[waterMask == waterFlag] = 1
      nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size=3)

      # Fire detection confidence test 17
      z4 = b22minusBG / b22MADfilt

      # Fire detection confidence test 18
      zDeltaT = (deltaTbgMask - deltaTmeanFilt) / deltaTMADFilt

      with np.errstate(invalid='ignore'):
        firesNclouds = nCloudAdj[(allFires == 1)]
        firesZ4 = z4[(allFires == 1)]
        firesZdeltaT = zDeltaT[(allFires == 1)]
        firesB22bgMask = b22bgMask[(allFires == 1)]
        firesNwater = nWaterAdj[(allFires == 1)]
        firesDayFlag = dayFlag[(allFires == 1)]

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

      # Detection confidence for the daytime
      confArrayDay = np.row_stack((C1day, C2, C3, C4, C5))
      detnConfDay = gmean(confArrayDay, axis=0)

      # Detection confidence for the nighttime
      confArrayNight = np.row_stack((C1night, C2, C3))
      detnConfNight = gmean(confArrayNight, axis=0)

      # Detection confidence for both day and night
      detnConf = np.zeros_like(detnConfDay, dtype=np.float)
      detnConf[firesDayFlag == 1] = detnConfDay[firesDayFlag == 1]
      detnConf[firesDayFlag == 0] = detnConfNight[firesDayFlag == 0]

      with np.errstate(invalid='ignore'):
        FRPx = np.where((allFires == 1))[1]
        FRPsample = FRPx + min1
        FRPy = np.where((allFires == 1))[0]
        FRPline = FRPy + min0
        FRPlats = croppedArrays['LAT'][(allFires == 1)]
        FRPlons = croppedArrays['LON'][(allFires == 1)]
        FRPT21 = croppedArrays['BAND22'][(allFires == 1)]
        FRPT31 = croppedArrays['BAND31'][(allFires == 1)]
        FRPMeanT21 = b22meanFilt[(allFires == 1)]
        FRPMeanT31 = b31meanFilt[(allFires == 1)]
        FRPMeanDT = deltaTmeanFilt[(allFires == 1)]
        FRPMADT21 = b22MADfilt[(allFires == 1)]
        FRPMADT31 = b31MADfilt[(allFires == 1)]
        FRP_MAD_DT = deltaTMADFilt[(allFires == 1)]
        FRP_AdjCloud = nCloudAdj[(allFires == 1)]
        FRP_AdjWater = nWaterAdj[(allFires == 1)]
        FRP_NumValid = nValid[(allFires == 1)]
        FRP_confidence = detnConf * 100
        FRPpower = FRP2016[(allFires == 1)]

      exportCSV = np.column_stack(
        [FRPline, FRPsample, FRPlats, FRPlons, FRPT21, FRPT31, FRPMeanT21, FRPMeanT31, FRPMeanDT, FRPMADT21, FRPMADT31,
         FRP_MAD_DT, FRPpower, FRP_AdjCloud, FRP_AdjWater, FRP_NumValid, FRP_confidence])

      exportCSV = [x for x in exportCSV if -4 not in x]

      if len(exportCSV) > 0:

        hdr = '"FRPline",' \
              '"FRPsample",' \
              '"FRPlats",' \
              '"FRPlons",' \
              '"FRPT21",' \
              '"FRPT31",' \
              '"FRPMeanT21",' \
              '"FRPMeanT31",' \
              '"FRPMeanDT",' \
              '"FRPMADT21",' \
              '"FRPMADT31",' \
              '"FRP_MAD_DT",' \
              '"FRPpower",' \
              '"FRP_AdjCloud",' \
              '"FRP_AdjWater",' \
              '"FRP_NumValid",' \
              '"FRP_confidence"'

        # Change the directory to the CSV output save directory
        os.chdir(cwd)

        np.savetxt(
          filMOD02.replace('hdf', '') + "csv", exportCSV, delimiter="\t", header=hdr,
          fmt=[
            "%d", # line
            "%d", # sample
            "%.5f", # lats
            "%.5f", # lons
            "%.2f", # t21
            "%.2f", # t31
            "%.2f", # mean t21
            "%.2f", # mean t31
            "%.2f", # mean dt
            "%.2f", # mad t21
            "%.2f", # mad t31
            "%.2f", # mad dt
            "%." + str(decimal) + "f", # power
            "%d", # cloud
            "%d", # water
            "%d", # valid
            "%.2f" # conf
          ]
        )

        # Change the directory back to the HDF loading directory - for the next function invocation via list comprehension
        os.chdir(hdfDir)

def run(directory, index, minLat, maxLat, minLon, maxLon, reductionFactor, minNcount, minNfrac, minKsize, maxKsize, decimalPlaces):

  # Current working directory in order to save the CSV output to the project files directory
  cwd = os.getcwd()
  # HDF loading directory
  hdfDir = directory + "/" + str(index)

  os.chdir(hdfDir)
  HDF03 = [hdf for hdf in os.listdir('.') if ".hdf" in hdf and "D03" in hdf]
  HDF02 = [hdf for hdf in os.listdir('.') if ".hdf" in hdf and "D02" in hdf]

  [process(hdf, HDF03, float(minLat), float(maxLat), float(minLon), float(maxLon),
           int(reductionFactor), int(minNcount), float(minNfrac), int(minKsize), int(maxKsize), int(decimalPlaces),
           cwd, hdfDir) for hdf in HDF02]