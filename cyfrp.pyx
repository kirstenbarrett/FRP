# cython: profile=True

from scipy import ndimage
import numpy as np
import os
from osgeo import gdal
import datetime
from scipy.stats import gmean
import math
# cimport numpy as np

def adjCloud(kernel):
  nghbors = kernel[range(0, 4) + range(5, 9)]
  cloudNghbors = kernel[np.where(nghbors == 1)]
  nCloudNghbr = len(cloudNghbors)
  return nCloudNghbr

def adjWater(kernel):
  nghbors = kernel[range(0, 4) + range(5, 9)]
  waterNghbors = kernel[np.where(nghbors == 1)]
  nWaterNghbr = len(waterNghbors)
  return nWaterNghbr

def makeFootprint(kSize):
  fpZeroLine = (kSize - 1) / 2
  fpZeroColStart = fpZeroLine - 1
  fpZeroColEnd = fpZeroColStart + 3
  fp = np.ones((kSize, kSize), dtype='int_')
  fp[fpZeroLine, fpZeroColStart:fpZeroColEnd] = -5
  return fp

def nValidFilt(kernel, kSize, minKsize, maxKsize):  # Use background mask files
  nghbrCnt = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    fpMask = makeFootprint(kSize)
    kernel[np.where(fpMask < 0)] = -5
    nghbrs = kernel[np.where(kernel > 0)]
    nghbrCnt = len(nghbrs)

  return nghbrCnt

def nRejectBGfireFilt(kernel, kSize, minKsize, maxKsize):
  nRejectBGfire = -4
  kernel = kernel.reshape((kSize, kSize))
  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    nRejectBGfire = len(kernel[np.where(kernel == -3)])

  return nRejectBGfire

def nRejectWaterFilt(kernel, kSize, minKsize, maxKsize):
  nRejectWater = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if (kSize == minKsize) | (centerVal == -4):
    nRejectWater = len(kernel[np.where(kernel == -1)])

  return nRejectWater

def nUnmaskedWaterFilt(kernel, kSize, minKsize, maxKsize):
  nUnmaskedWater = -4
  kernel = kernel.reshape((kSize, kSize))

  centerVal = kernel[((kSize - 1) / 2), ((kSize - 1) / 2)]

  if ((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-3, 0))):
    nUnmaskedWater = len(kernel[np.where(kernel == -6)])

  return nUnmaskedWater

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

def wakelinMeanMADFilter(band, maxKsize, minKsize, minNcount, minNfrac):
  bSize = (maxKsize - 1) / 2
  bandMatrix = np.pad(band, ((bSize, bSize), (bSize, bSize)), mode='symmetric')

  bandFiltsMean2 = {}
  bandFiltsMAD2 = {}
  kSize = minKsize
  i, j = np.shape(band)

  while kSize <= maxKsize:

    bandMADFilt2_tmp = np.full([i, j], -4.0)
    bandMeanFilt2_tmp = np.full([i, j], -4.0)

    halfK = (kSize - 1) / 2
    for x in range(bSize, i + bSize):
      for y in range(bSize, j + bSize):

        xmhk = x - halfK
        xphk = x + halfK + 1
        ymhk = y - halfK
        yphk = y + halfK + 1

        # Must copy kernel otherwise it is a reference to original array - hence original is changed!
        kernel = bandMatrix[xmhk:xphk:1, ymhk:yphk:1].copy()
        centerVal = bandMatrix[x, y]

        if ((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2, 0))):
          fpMask = makeFootprint(kSize)
          kernel[np.where(fpMask < 0)] = -5
          nghbrs = kernel[np.where(kernel > 0)]
          nghbrCnt = len(nghbrs)

          if (nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * (kSize ** 2))):
            bgMean = np.mean(nghbrs)
            meanDists = np.abs(nghbrs - bgMean)
            bgMAD = np.mean(meanDists)

            # Remember - Results matrix is smaller than padded dataset by bSize in all directions
            xmb = x - bSize
            ymb = y - bSize
            bandMADFilt2_tmp[xmb, ymb] = bgMAD
            bandMeanFilt2_tmp[xmb, ymb] = bgMean

    filtNameMean2 = 'bandFiltMean' + str(kSize)
    bandFiltsMean2[filtNameMean2] = bandMeanFilt2_tmp
    filtNameMAD2 = 'bandFiltMAD' + str(kSize)
    bandFiltsMAD2[filtNameMAD2] = bandMADFilt2_tmp

    kSize += 2

  bandFiltMean2 = bandFiltsMean2['bandFiltMean' + str(minKsize)]
  bandFiltMAD2 = bandFiltsMAD2['bandFiltMAD' + str(minKsize)]
  kSize = minKsize + 2

  while kSize <= maxKsize:
    bandFiltMean2[np.where(bandFiltMean2 == -4)] = bandFiltsMean2['bandFiltMean' + str(kSize)][
      np.where(bandFiltMean2 == -4)]
    bandFiltMAD2[np.where(bandFiltMAD2 == -4)] = bandFiltsMAD2['bandFiltMAD' + str(kSize)][np.where(bandFiltMAD2 == -4)]
    kSize += 2

  return bandFiltMean2, bandFiltMAD2

def wakelinMeanFilter(band, maxKsize, minKsize, minNcount, minNfrac):
  # Add boundary for largest known tile size (maxKsize)
  bSize = (maxKsize - 1) / 2
  bandMatrix = np.pad(band, ((bSize, bSize), (bSize, bSize)), mode='symmetric')

  bandFiltsMean2 = {}
  kSize = minKsize
  i, j = np.shape(band)

  # Loop through dataset
  while kSize <= maxKsize:
    bandMeanFilt2_tmp = np.full([i, j], -4.0)
    halfK = (kSize - 1) / 2
    for x in range(bSize, i + bSize):
      for y in range(bSize, j + bSize):

        xmhk = x - halfK
        xphk = x + halfK + 1
        ymhk = y - halfK
        yphk = y + halfK + 1

        # Must copy kernel otherwise it is a reference to original array - hence original is changed!
        kernel = bandMatrix[xmhk:xphk:1, ymhk:yphk:1].copy()
        centerVal = bandMatrix[x, y]

        if ((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2, 0))):
          fpMask = makeFootprint(kSize)
          kernel[np.where(fpMask < 0)] = -5
          nghbrs = kernel[np.where(kernel > 0)]
          nghbrCnt = len(nghbrs)

          if (nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * (kSize ** 2))):
            bgMean = np.mean(nghbrs)

            # Remember - Results matrix is smaller than padded dataset by bSize in all directions
            xmb = x - bSize
            ymb = y - bSize
            bandMeanFilt2_tmp[xmb, ymb] = bgMean

    filtNameMean2 = 'bandFiltMean' + str(kSize)
    bandFiltsMean2[filtNameMean2] = bandMeanFilt2_tmp
    kSize += 2

  bandFiltMean2 = bandFiltsMean2['bandFiltMean' + str(minKsize)]
  kSize = minKsize + 2

  while kSize <= maxKsize:
    bandFiltMean2[np.where(bandFiltMean2 == -4)] = bandFiltsMean2['bandFiltMean' + str(kSize)][
      np.where(bandFiltMean2 == -4)]
    kSize += 2

  return bandFiltMean2

def wakelinMADFilter(band, maxKsize, minKsize, minNcount, minNfrac):
  # Add boundary for largest known tile size (maxKsize)
  bSize = (maxKsize - 1) / 2
  bandMatrix = np.pad(band, ((bSize, bSize), (bSize, bSize)), mode='symmetric')

  bandFiltsMAD2 = {}
  kSize = minKsize
  i, j = np.shape(band)

  # Loop through dataset
  while kSize <= maxKsize:

    bandMADFilt2_tmp = np.full([i, j], -4.0)

    halfK = (kSize - 1) / 2
    for x in range(bSize, i + bSize):
      for y in range(bSize, j + bSize):

        xmhk = x - halfK
        xphk = x + halfK + 1
        ymhk = y - halfK
        yphk = y + halfK + 1

        # Must copy kernel otherwise it is a reference to original array - hence original is changed!
        kernel = bandMatrix[xmhk:xphk:1, ymhk:yphk:1].copy()
        centerVal = bandMatrix[x, y]

        if ((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2, 0))):
          fpMask = makeFootprint(kSize)
          kernel[np.where(fpMask < 0)] = -5
          nghbrs = kernel[np.where(kernel > 0)]
          nghbrCnt = len(nghbrs)

          if (nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * (kSize ** 2))):
            bgMean = np.mean(nghbrs)
            meanDists = np.abs(nghbrs - bgMean)
            bgMAD = np.mean(meanDists)

            # Remember - result matrix is smaller than padded data set by bSize in all directions
            xmb = x - bSize
            ymb = y - bSize
            bandMADFilt2_tmp[xmb, ymb] = bgMAD

    filtNameMAD2 = 'bandFiltMAD' + str(kSize)
    bandFiltsMAD2[filtNameMAD2] = bandMADFilt2_tmp

    kSize += 2

  bandFiltMAD2 = bandFiltsMAD2['bandFiltMAD' + str(minKsize)]
  kSize = minKsize + 2

  while kSize <= maxKsize:
    bandFiltMAD2[np.where(bandFiltMAD2 == -4)] = bandFiltsMAD2['bandFiltMAD' + str(kSize)][np.where(bandFiltMAD2 == -4)]
    kSize += 2

  return bandFiltMAD2

def process(filMOD02, HDF03, minLat, maxLat, minLon, maxLon, reductionFactor, minNcount, minNfrac, minKsize, maxKsize):

  b22saturationVal = 331
  increaseFactor = 1 + (1 - reductionFactor)
  waterFlag = -1
  cloudFlag = -2
  bgFlag = -3
  resolution = 5
  datsWdata = []

  # Coefficients for radiance calculations
  coeff1 = 119104200
  coeff2 = 14387.752
  lambda21and22 = 3.959
  lambda31 = 11.009
  lambda32 = 12.02

  # Layers for reading in HDF files
  layersMOD02 = ['EV_1KM_Emissive', 'EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB']
  layersMOD03 = ['Land/SeaMask', 'Latitude', 'Longitude', 'SolarAzimuth', 'SolarZenith', 'SensorAzimuth', 'SensorZenith']

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

    allArrays = {}  # Clipped to min/max lat/long
    for b in fullArrays.keys():
      cropB = fullArrays[b][min0:max0, min1:max1]
      allArrays[b] = cropB

    [nRows, nCols] = np.shape(allArrays['BAND22'])

    # Test for b22 saturation
    allArrays['BAND22'][np.where(allArrays['BAND22'] >= b22saturationVal)] = allArrays['BAND21'][
      np.where(allArrays['BAND22'] >= b22saturationVal)]

    # Day/Night flag
    dayFlag = np.zeros((nRows, nCols), dtype=np.int)
    dayFlag[np.where(allArrays['SolarZenith'] < 8500)] = 1

    # Create water mask
    waterMask = np.zeros((nRows, nCols), dtype=np.int)
    waterMask[np.where(allArrays['LANDMASK'] != 1)] = waterFlag

    # Crate cloud mask
    cloudMask = np.zeros((nRows, nCols), dtype=np.int)
    cloudMask[((allArrays['BAND1x1k'] + allArrays['BAND2x1k']) > 900)] = cloudFlag
    cloudMask[(allArrays['BAND32'] < 265)] = cloudFlag
    cloudMask[((allArrays['BAND1x1k'] + allArrays['BAND2x1k']) > 700) & (allArrays['BAND32'] < 285)] = cloudFlag

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
    b22meanFilt, b22MADfilt = wakelinMeanMADFilter(b22bgMask, maxKsize, minKsize, minNcount, minNfrac)
    b22minusBG = np.copy(b22CloudWaterMasked) - np.copy(b22meanFilt)
    b31meanFilt, b31MADfilt = wakelinMeanMADFilter(b31bgMask, maxKsize, minKsize, minNcount, minNfrac)
    deltaTmeanFilt, deltaTMADFilt = wakelinMeanMADFilter(deltaTbgMask, maxKsize, minKsize, minNcount, minNfrac)

    b22bgRej = np.copy(allArrays['BAND22'])
    b22bgRej[np.where(bgMask != bgFlag)] = bgFlag
    b22rejMeanFilt, b22rejMADfilt = wakelinMeanMADFilter(b22bgRej, maxKsize, minKsize, minNcount, minNfrac)

    # Potential fire test
    potFire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      potFire[(dayFlag == 1) & (allArrays['BAND22'] > (310 * reductionFactor)) & (deltaT > (10 * reductionFactor)) & (
        allArrays['BAND2x1k'] < (300 * increaseFactor))] = 1
      potFire[(dayFlag == 0) & (allArrays['BAND22'] > (305 * reductionFactor)) & (deltaT > (10 * reductionFactor))] = 1

    # Absolute threshold test for removing sun glint
    absValTest = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      absValTest[(dayFlag == 1) & (allArrays['BAND22'] > (360 * reductionFactor))] = 1
      absValTest[(dayFlag == 0) & (allArrays['BAND22'] > (305 * reductionFactor))] = 1

    # Context fire test 2
    deltaTMADfire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      deltaTMADfire[deltaT > (deltaTmeanFilt + (3.5 * deltaTMADFilt))] = 1

    # Context fire test 3
    deltaTfire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      deltaTfire[np.where(deltaT > (deltaTmeanFilt + 6))] = 1

    # Context fire test 4
    B22fire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      B22fire[(b22CloudWaterMasked > (b22meanFilt + (3 * b22MADfilt)))] = 1

    # Context fire test 5
    B31fire = np.zeros((nRows, nCols), dtype=np.int)
    B31fire[(b31CloudWaterMasked > (b31meanFilt + b31MADfilt - 4))] = 1

    # Context fire test 6
    B22rejFire = np.zeros((nRows, nCols), dtype=np.int)

    with np.errstate(invalid='ignore'):
      B22rejFire[(b22rejMADfilt > 5)] = 1

    # Combine tests to create tentative fires
    fireLocTentative = deltaTMADfire * deltaTfire * B22fire

    fireLocB31andB22rejFire = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      fireLocB31andB22rejFire[np.where((B22rejFire == 1) | (B31fire == 1))] = 1
    fireLocTentativeDay = potFire * fireLocTentative * fireLocB31andB22rejFire

    dayFires = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dayFires[(dayFlag == 1) & ((absValTest == 1) | (fireLocTentativeDay == 1))] = 1

    # Nighttime definite fire tests
    nightFires = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      nightFires[((dayFlag == 0) & ((fireLocTentative == 1) | absValTest == 1))] = 1

    # Sun glint rejection
    relAzimuth = allArrays['SensorAzimuth'] - allArrays['SolarAzimuth']
    cosThetaG = (np.cos(allArrays['SensorZenith']) * np.cos(allArrays['SolarZenith'])) - (
      np.sin(allArrays['SensorZenith']) * np.sin(allArrays['SolarZenith']) * np.cos(relAzimuth))
    thetaG = np.arccos(cosThetaG)
    thetaG = (thetaG / 3.141592) * 180

    # Sun glint test 8
    sgTest8 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest8[np.where(thetaG < 2)] = 1

    # Sun glint test 9
    sgTest9 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest9[np.where((thetaG < 8) & (allArrays['BAND1x1k'] > 100) & (allArrays['BAND2x1k'] > 200) & (
        allArrays['BAND7x1k'] > 120))] = 1

    # Sun glint test 10
    waterLoc = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      waterLoc[np.where(waterMask == waterFlag)] = 1
    nWaterAdj = ndimage.generic_filter(waterLoc, adjWater, size=3)
    nRejectedWater = runFilt(waterMask, nRejectWaterFilt, minKsize, maxKsize)
    with np.errstate(invalid='ignore'):
      nRejectedWater[np.where(nRejectedWater < 0)] = 0

    sgTest10 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgTest10[np.where((thetaG < 12) & ((nWaterAdj + nRejectedWater) > 0))] = 1

    sgAll = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      sgAll[(sgTest8 == 1) | (sgTest9 == 1) | (sgTest10 == 1)] = 1

    # Desert boundary rejection
    nValid = runFilt(b22bgMask, nValidFilt, minKsize, maxKsize)
    nRejectedBG = runFilt(bgMask, nRejectBGfireFilt, minKsize, maxKsize)

    with np.errstate(invalid='ignore'):
      nRejectedBG[np.where(nRejectedBG < 0)] = 0

    # Desert boundary test 11
    dbTest11 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest11[np.where(nRejectedBG > (0.1 * nValid))] = 1

    # Desert boundary test 12
    dbTest12 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest12[(nRejectedBG >= 4)] = 1

    # Desert boundary test 13
    dbTest13 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest13[np.where(allArrays['BAND2x1k'] > 150)] = 1

    # Desert boundary test 14
    dbTest14 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest14[(b22rejMeanFilt < 345)] = 1

    # Desert boundary test 15
    dbTest15 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest15[(b22rejMADfilt < 3)] = 1

    # Desert boundary test 16
    dbTest16 = np.zeros((nRows, nCols), dtype=np.int)
    with np.errstate(invalid='ignore'):
      dbTest16[(b22CloudWaterMasked < (b22rejMeanFilt + (6 * b22rejMADfilt)))] = 1

    # Reject anything that fulfills desert boundary criteria
    dbAll = dbTest11 * dbTest12 * dbTest13 * dbTest14 * dbTest15 * dbTest16

    # Coastal false alarm rejection
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

    if np.max(allFires) > 0:
      datsWdata.append(t)

      b22firesAllMask = allFires * allArrays['BAND22']
      b22bgAllMask = allFires * b22meanFilt

      b22maskEXP = b22firesAllMask.astype(float) ** 8
      b22bgEXP = b22bgAllMask.astype(float) ** 8

      frpMW = 4.34 * (10 ** (-19)) * (b22maskEXP - b22bgEXP)  # AREA TERM HERE

      frpMWabs = frpMW * potFire

      # Detection confidence
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

      # Fire detection confidence test 22
      C4 = 1 - rampFn(firesNclouds, 0, 6)

      # Fire detection confidence test 23
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
      np.savetxt(filMOD02.replace('hdf', '') + 'frp20160512_hdf_hps.csv', exportCSV, delimiter=",", header=hdr)


def run(datapath, procid, minLat, maxLat, minLon, maxLon, reductionFactor, minNcount, minNfrac, minKsize, maxKsize):

  HDF03 = [hdf for hdf in os.listdir('.') if ".hdf" in hdf and "D03" in hdf]
  HDF02 = [hdf for hdf in os.listdir('.') if ".hdf" in hdf and "D02" in hdf]
  [process(hdf, HDF03, minLat, maxLat, minLon, maxLon, reductionFactor, minNcount, minNfrac, minKsize, maxKsize) for hdf in HDF02]

run('', 0, 65, 65.525, -148, -146, 1, 8, 0.25, 5, 21)