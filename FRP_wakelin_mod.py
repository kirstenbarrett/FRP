#import cython
from scipy import ndimage
import numpy as np
import os
from osgeo import gdal
import time

#Geometric settings for sampling
minNcount = 8
minNfrac = 0.25
minKsize = 5
maxKsize = 21
b21saturationVal = 450 #???

filList = os.listdir('.')
filNam = 'MOD021KM.A2004178.2120.005.'
bands = ['BAND1','BAND2','BAND7','BAND21','BAND22','BAND31','BAND32','LANDMASK','SolarZenith','SolarAzimuth','SensorZenith','SensorAzimuth','LAT','LON']

allArrays = {}
for b in bands:
    fullFilName = filNam + b + '.tif'
    ds = gdal.Open(fullFilName)
    data = np.array(ds.GetRasterBand(1).ReadAsArray())
    data = data[1472:1546,566:656] #BOUNDARY FIRE AREA
 #   data = data[1105:2029,84:1065] #BOREAL AK AREA

    if b == 'BAND21' or b == 'BAND22' or b == 'BAND31' or b == 'BAND32':
#        data = np.int_(np.rint(data))
        data = data
    if b == 'BAND1' or b == 'BAND2' or b == 'BAND7':
        b = b + 'x1k'
        data = np.int_(np.rint(data*1000))

    allArrays[b] = data

[nRows,nCols] = np.shape(allArrays['BAND21'])

#DAY/NIGHT FLAG
dayFlag = np.zeros((nRows,nCols),dtype=np.int)
dayFlag[np.where(allArrays['SolarZenith'] < 8500)] = 1

waterFlag = -1
cloudFlag = -2

#CREATE WATER MASK
waterMask = np.zeros((nRows,nCols),dtype=np.int)
waterMask[np.where(allArrays['LANDMASK']!=1)] = waterFlag

#CREATE CLOUD MASK (SET DATATYPE)
cloudMask =np.zeros((nRows,nCols),dtype=np.int)
cloudMask[((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>900)] = cloudFlag
cloudMask[(allArrays['BAND32']<265)] = cloudFlag
cloudMask[((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>700)&(allArrays['BAND32']<285)] = cloudFlag

#MASK CLOUDS AND WATER FROM INPUT BANDS
b21CloudWaterMasked = np.copy(allArrays['BAND21'])
b21CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
b21CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

b22CloudWaterMasked = np.copy(allArrays['BAND22'])
b22CloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
b22CloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

b31CloudWaterMasked = np.copy(allArrays['BAND31'])
b31CloudWaterMasked [np.where(waterMask == waterFlag)] = waterFlag
b31CloudWaterMasked [np.where(cloudMask == cloudFlag)] = cloudFlag

deltaT = np.abs(allArrays['BAND21'] - allArrays['BAND31'])
deltaTCloudWaterMasked = np.copy(deltaT)
deltaTCloudWaterMasked[np.where(waterMask == waterFlag)] = waterFlag
deltaTCloudWaterMasked[np.where(cloudMask == cloudFlag)] = cloudFlag

#CREATE A MASK FOR BACKGROUND SAMPLING
bgFlag = -3
bgMask = np.zeros((nRows,nCols),dtype=np.int)

with np.errstate(invalid='ignore'):
    bgMask[np.where((dayFlag == 1) & (allArrays['BAND21'] >325) & (deltaT >20))] = bgFlag
    bgMask[np.where((dayFlag == 0) & (deltaT >310)& (deltaT >10))] = bgFlag

    b21bgMask = np.copy(b21CloudWaterMasked)
    b21bgMask[np.where(bgMask == bgFlag)] = bgFlag

    b22bgMask = np.copy(b22CloudWaterMasked)
    b22bgMask[np.where(bgMask == bgFlag)] = bgFlag

    b31bgMask = np.copy(b31CloudWaterMasked)
    b31bgMask[np.where(bgMask == bgFlag)] = bgFlag

    deltaTbgMask = np.copy(deltaTCloudWaterMasked)
    deltaTbgMask[np.where(bgMask == bgFlag)] = bgFlag


#OMIT SCANLINE NEIGHBORS FROM SAMPLING
def makeFootprint(kSize):
    fpZeroLine = (kSize-1)/2
    fpZeroColStart = fpZeroLine-1
    fpZeroColEnd = fpZeroColStart+3
    fp = np.ones((kSize,kSize),dtype = 'int_')
    fp[fpZeroLine,fpZeroColStart:fpZeroColEnd] = -5

    return fp


#RETURN MEAN OF NON-BACKGROUND FIRE NEIGHBORS
def meanFilt(kernel,kSize,minKsize,maxKsize):
    bgMean = -4
    kernel = kernel.reshape((kSize,kSize))

    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
#-    print "Kernel A", centerVal
#-    print kernel
    if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
        fpMask = makeFootprint(kSize)
        kernel[np.where(fpMask < 0)] = -5
        nghbrs = kernel[np.where(kernel > 0)]
        nghbrCnt = len(nghbrs)
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
            bgMean = np.mean(nghbrs)

    return bgMean

#RETURN MEAN ABSOLUTE DEVIATION OF NON-BACKGROUND FIRE NEIGHBORS
def MADfilt(kernel,kSize,minKsize,maxKsize):
    bgMAD = -4
    kernel = kernel.reshape((kSize,kSize))

    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]

    if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
        fpMask = makeFootprint(kSize)
        kernel[np.where(fpMask < 0)] = -5
        nghbrs = kernel[np.where(kernel > 0)]
        nghbrCnt = len(nghbrs)
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
            bgMean = np.mean(nghbrs)
            meanDists = np.abs(nghbrs - bgMean)
            bgMAD = np.mean(meanDists)

    return bgMAD

#RUNS FILTERS ON PROGRESSIVELY LARGER KERNEL SIZES, COMBINES RESULTS FROM SMALLEST KSIZE
def runFilt(band,filtFunc,minKsize,maxKsize):
    filtBand = band
    kSize = minKsize
    bandFilts = {}
    
    while kSize <=  maxKsize:
        filtName = 'bandFilt'+str(kSize)
        filtBand = ndimage.generic_filter(filtBand, filtFunc, size = kSize, extra_arguments= (kSize,minKsize,maxKsize))
        bandFilts[filtName] = filtBand
        kSize += 2
        
    bandFilt = bandFilts['bandFilt'+str(minKsize)]
    kSize = minKsize + 2
    
    while kSize <= maxKsize:
        bandFilt[np.where(bandFilt == -4)] = bandFilts['bandFilt'+str(kSize)][np.where(bandFilt == -4)]
        kSize += 2
        
    return bandFilt

####CONTEXT FIRE TEST 2:
print "Start1", time.time()
print "Delta Tmean matrix"
deltaTmeanFilt = runFilt(deltaTbgMask,meanFilt,minKsize,maxKsize)


###+print deltaTmeanFilt
##
######deltaT MAD Filtering
##print "Delta TMAD matrix"
##deltaTMADFilt = runFilt(deltaTbgMask,MADfilt,minKsize,maxKsize)
##
##print "Finish", time.time()
##
###-----------------------------------------------------------------
##
##print "Start 2", time.time()

#WAKELIN MODS (B21mean,B21/22MAD, B22mean (if necessary), deltaTmean, deltaTMAD
# Add boundary for largest known tile size (maxKsize)
bSize = (maxKsize-1)/2 
deltaMatrix = np.pad(deltaTbgMask,((bSize,bSize),(bSize,bSize)),mode='symmetric')
b21Matrix = np.pad(b21bgMask,((bSize,bSize),(bSize,bSize)),mode='symmetric')
b22Matrix = np.pad(b22bgMask,((bSize,bSize),(bSize,bSize)),mode='symmetric')

# Initialize stuff needed in loop
bandFiltsMean2 = {}
bandFiltsMAD2 = {}
kSize = minKsize 
i,j = np.shape(deltaTbgMask)

# Loop through dataset
while kSize <=  maxKsize:
    
    deltaTMADFilt2_tmp  = np.full([i,j], -4.0)
    deltaTmeanFilt2_tmp = np.full([i,j], -4.0)
    
#-    print "B Filter", kSize
    halfK = (kSize-1)/2
    for x in range(bSize,i+bSize):
        for y in range(bSize,j+bSize):

            xmhk = x-halfK
            xphk = x+halfK+1
            ymhk = y-halfK
            yphk = y+halfK+1

            # Must copy kernel otherwise it is a reference to original array - hence original is changed!
            kernel = deltaMatrix[xmhk:xphk:1,ymhk:yphk:1].copy()
            centerVal = deltaMatrix[x,y]

#-            print "Tile = ",xmhk, xphk, ymhk, yphk
#-           print "Center = ", x, y
#-            print "CenterVal = ", centerVal
#-            print kernel
            if (((kSize == minKsize) | (centerVal == -4)) & (centerVal not in (range(-2,0)))):
                fpMask = makeFootprint(kSize)
                kernel[np.where(fpMask < 0)] = -5
                nghbrs = kernel[np.where(kernel > 0)]
                nghbrCnt = len(nghbrs)
                
                if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
                    bgMean = np.mean(nghbrs)
                    meanDists = np.abs(nghbrs - bgMean)
                    bgMAD = np.mean(meanDists)

                    # Remember - Results matrix is smaller than padded dataset by bSize in all directions
                    xmb = x-bSize
                    ymb = y-bSize
                    deltaTMADFilt2_tmp[xmb,ymb] = bgMAD
                    deltaTmeanFilt2_tmp[xmb,ymb] = bgMean
                    #deltaTMADFilt2_tmp[xmhk,ymhk] = bgMAD
                    #deltaTmeanFilt2_tmp[xmhk,ymhk] = bgMean
                    
    filtNameMean2 = 'bandFiltMean'+str(kSize)
    bandFiltsMean2[filtNameMean2] = deltaTmeanFilt2_tmp
    filtNameMAD2 = 'bandFiltMAD'+str(kSize)
    bandFiltsMAD2[filtNameMAD2] = deltaTMADFilt2_tmp

    kSize += 2



bandFiltMean2 = bandFiltsMean2['bandFiltMean'+str(minKsize)]
bandFiltMAD2 = bandFiltsMAD2['bandFiltMAD'+str(minKsize)]
kSize = minKsize + 2

while kSize <= maxKsize:
    bandFiltMean2[np.where(bandFiltMean2 == -4)] = bandFiltsMean2['bandFiltMean'+str(kSize)][np.where(bandFiltMean2 == -4)]
    bandFiltMAD2[np.where(bandFiltMAD2 == -4)] = bandFiltsMAD2['bandFiltMAD'+str(kSize)][np.where(bandFiltMAD2 == -4)]
    kSize += 2


##print "Delta Tmean matrix"
##print bandFilts2['bandFilt5']
##print "=================================="
##print bandFilts2['bandFilt7']
##print "=================================="
##print bandFilts2['bandFilt9']
print "Finish 2", time.time()
