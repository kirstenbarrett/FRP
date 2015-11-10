#Just needs multi-dimensional footprint
#and the sunglint mask...
from scipy import ndimage
import numpy as np
import os
from osgeo import gdal
from collections import namedtuple

os.chdir('/Users/kirsten/Documents/data/MODIS/Boundary_swaths')
filList = os.listdir('.')
filNam = 'MOD021KM.A2004178.2120.005.'
bands = ['BAND1','BAND2','BAND21','BAND22','BAND31','BAND32','landmask','SolarZenith','LAT','LON']

#READ IN AL REFLECTANCE AND EMITTED BANDS
allArrays = {}
for b in bands:
    fullFilName = filNam + b + '.tif'
    ds = gdal.Open(fullFilName)
    data = np.array(ds.GetRasterBand(1).ReadAsArray())

    if b == 'BAND21' or b == 'BAND22' or b == 'BAND31' or b == 'BAND32':
        data = np.int_(np.rint(data))
    if b == 'BAND1' or b == 'BAND2':
        b = b + 'x1k'
        data = np.int_(np.rint(data*1000))

    allArrays[b] = data

[nRows,nCols] = np.shape(allArrays['BAND21'])

#DAY/NIGHT FLAG
dayFlag = np.zeros((nRows,nCols),dtype=np.int)
dayFlag[np.where(allArrays['SolarZenith'] < 8500)] = 1

#CHANGES LANDMASK TO BINARY (SET DATATYPE) "LAND" = 1
waterMask = np.ones((nRows,nCols),dtype=np.int)
waterMask[np.where(allArrays['landmask']>1)] = 0

#CREATE CLOUD MASK (SET DATATYPE) "CLOUD" = 0
b1plus2gt900=np.ones((nRows,nCols),dtype=np.int)
b1plus2gt900[np.where((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>900)] = 0

b32lt265 = np.ones((nRows,nCols),dtype=np.int)
b32lt265[np.where(allArrays['BAND32']<265)] = 0

##MASK CLOUDS AND WATER
b1plus2gt700=np.ones((nRows,nCols),dtype=np.int)
b1plus2gt700[np.where((allArrays['BAND1x1k']+allArrays['BAND2x1k'])>700)] = 0

b32lt285=np.ones((nRows,nCols),dtype=np.int)
b32lt285[np.where(allArrays['BAND32']<285)]=0

temp=np.ones((nRows,nCols),dtype=np.int)
temp[np.where(b1plus2gt700+b32lt265)==0] = 0

cloudMask = b1plus2gt900*b32lt285*temp

maskWaterCloud = waterMask*cloudMask


b21mask = allArrays['BAND21']*maskWaterCloud
b22mask = allArrays['BAND22']*maskWaterCloud
b31mask = allArrays['BAND31']*maskWaterCloud

deltaT = allArrays['BAND21'] - allArrays['BAND31'] + 1 #ADD ONE SO ALL REAL ZEROS ARE MAINTAINED
deltaT *= maskWaterCloud

##########################
##AFTER ALL THE DATA HAVE BEEN READ IN
##########################

minNcount = 8
minNfrac = 0.25
minKsize = 5
maxKsize = 21
b21saturationVal = 450 #???

bgMask = np.ones((nRows,nCols))
bgMask[np.where((dayFlag == 1) & (b21mask >325))] = 0
bgMask[np.where((dayFlag == 1) & (deltaT >20))] = 0
bgMask[np.where((dayFlag == 0) & (deltaT >310))] = 0
bgMask[np.where((dayFlag == 0) & (deltaT >10))] = 0

b21bgMask = b21mask*bgMask

def makeFootprint(kSize):
    fpZeroLine = (kSize-1)/2
    fpZeroColStart = fpZeroLine-1
    fpZeroColEnd = fpZeroColStart+3
    fp = np.ones((kSize,kSize),dtype = 'int_')
    fp[fpZeroLine,fpZeroColStart:fpZeroColEnd] = 0

    return fp

def meanFilt(kernel):
    bgMean = -1
    kernel = kernel.reshape((kSize,kSize))
    
    centerVal = kernel[((kSize-1)/2),((kSize-1)/2)]
    
    if ((kSize == 5) | (centerVal == -1)):
        fpMask = makeFootprint(kSize)
        kFpMask = kernel * fpMask
        nghbrCnt = np.count_nonzero(kFpMask)
        
        if ((nghbrCnt > minNcount) & (nghbrCnt > (minNfrac * ((kSize **2))))):
            bgMean = np.int_(np.rint(np.mean(kFpMask[np.nonzero(kFpMask)])))
            
    return bgMean

smplData = np.array([[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1],[2,1,2,4,5,6,1,6,7,8,1,3,5,6,2,5,1,6,1,2,1]])

kSize = 5
b21bgMask5x5 = ndimage.generic_filter(smplData, meanFilt, kSize)

kSize = 7
b21bgMask7x7 = ndimage.generic_filter(b21bgMask5x5, meanFilt, kSize)

kSize = 9
b21bgMask9x9 = ndimage.generic_filter(b21bgMask7x7 , meanFilt, kSize)

kSize = 11
b21bgMask11x11 = ndimage.generic_filter(b21bgMask9x9, meanFilt, kSize)

kSize = 13
b21bgMask13x13 = ndimage.generic_filter(b21bgMask11x11, meanFilt, kSize)



#REFL TIMES BG
