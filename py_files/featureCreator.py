import math

from  featuringFunctions import *


def windowingPattern(data,window_size,func):
  featureData=[0]*(window_size-1)
  for i in range(window_size-1,len(data)):
    featureData.append(func(data[i-window_size+1:i+1]))

  return featureData




def getFeature(data,window_size,feature):
  feature_function_map={
    'iemg':sumOfAbs,
    'mav':meanOfAbs,
    'mav1':meanOfWeightedAbs,
    'mav2':meanOfMultiWeightedAbs,
    'ssi':sumOfSquares
  }

  return (lambda data,window_size:windowingPattern(data,window_size,feature_function_map[feature]))(data,window_size)


data=[0,1,2,3,4,5,6,7,8,9]
window_size=3
a=getFeature(data,window_size,'iemg')
print(a)
