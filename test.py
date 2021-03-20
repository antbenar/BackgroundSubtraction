import numpy as np
from numpy import argmax, histogram
import tensorflow as tf
import os, sys
from generateData import GenerateData
from keras.preprocessing import image as kImage
import matplotlib.pyplot as plt

# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  Para configuracion de GPU
# =============================================================================
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# =============================================================================
# Few frames, load into memory directly
# =============================================================================
#maxW = 576
#maxH = 720
maxW = 320
maxH = 240

def threshold(img, thd):
    #maxShape = (576, 720, 1)
    img[img >= thd] = 1
    img[img < thd] = 0
    return img

def imshow(img):
    plt.imshow(img/255)
    plt.show()

def showImage(img):
    x = img
    x = kImage.array_to_img(x)
    x.show()

def numeric_score(prediction, groundtruth):
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    numDecimales = 3
    #FP = round(FP,numDecimales)
    #FN = round(FN,numDecimales)
    #TP = round(TP,numDecimales)
    #TN = round(TN,numDecimales)

    print(FP, ' ', FN, ' ', TP, ' ', TN)
    FMEASURE = round( 2*TP/(2*TP+FP+FN), numDecimales)
    PWC = round( 100*(FN+FP)/(TP+FN+FP+TN), numDecimales)
    return FP, FN, TP, TN, FMEASURE, PWC


def otsu2( hist, total ):
	"""
	This is the more common (optimized) implementation of otsu algorithm, the one you see on Wikipedia pages
	"""
	no_of_bins = len( hist ) # should be 256

	sum_total = 0
	for x in range( 0, no_of_bins ):
		sum_total += x * hist[x]
	
	weight_background 	  = 0.0
	sum_background 		  = 0.0
	inter_class_variances = []

	for threshold in range( 0, no_of_bins ):
		# background weight will be incremented, while foreground's will be reduced
		weight_background += hist[threshold]
		if weight_background == 0 :
			continue

		weight_foreground = total - weight_background
		if weight_foreground == 0 :
			break

		sum_background += threshold * hist[threshold]
		mean_background = sum_background / weight_background
		mean_foreground = (sum_total - sum_background) / weight_foreground

		inter_class_variances.append( weight_background * weight_foreground * (mean_background - mean_foreground)**2 )

	# find the threshold with maximum variances between classes
	return argmax(inter_class_variances)

# =============================================================================
# Main func
# =============================================================================

dataset = {
            'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
            'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
            'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass']
}
"""
dataset = {
            'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
            'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
            'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
            'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
            'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
            'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
            'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
            'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
}
"""

# =============================================================================
#--------------------------VARIABES GLOBALES------------------------------
method_name = 'LSTM3DCNN'
thd = 0.8 #threshold
main_dir = os.path.join('..')
mdl_dir = os.path.join('TrainResults')
# =============================================================================


###TESTING SPECIFIC FRAME
def testSpecificFrame(nameh5weights, category, scene, numImg):
    weightsDir =  os.path.join(mdl_dir, nameh5weights)
    model = tf.keras.models.load_model(weightsDir, custom_objects={'loss': 'binary_crossentropy'}, compile=False)
    model.summary()
    train_dir = os.path.join(main_dir, 'CDnet2014_dataset', category, scene, 'groundtruth')
    dataset_dir = os.path.join(main_dir, 'CDnet2014_dataset', category, scene, 'input')
    testData = generateData.generate(train_dir, dataset_dir, numImg, numImg+5)
    
    print ('Testing frame->>> ' + category + ' / ' + scene + ' / ' + str(numImg))
    result = model.predict(testData[0],verbose = 1)
    
    
    #-----show images
    #---Input
    showImage(testData[0][0][4])
    #---Groundtruth
    showImage(testData[1][0][4])
    #--output
    showImage(result[0][4])
    #showImage(threshold(result[0][4], 0.75))
    
    return result[0][4], testData[1][0][4], testData[0][0][4]

generateData = GenerateData(maxW, maxH)
#nameh5weights = 'LSTM_highway2epochs.h5'
#nameh5weights = 'LSTM_highway60epochs.h5'
#nameh5weights = 'LSTM_highway50epochs800frames.h5'

#"""
nameh5weights = 'LSTM50epochs200frames/mdl_pedestrians.h5'
category = 'baseline'
scene = 'pedestrians'
#numImg = 300
numImg = 200
#"""
"""nameh5weights = 'LSTM50epochs200frames/mdl_canoe.h5'
category = 'dynamicBackground'
scene = 'canoe'
numImg = 120"""
prediction, groundtruth, input = testSpecificFrame(nameh5weights, category, scene, numImg)

#otsu threshold
"""
hist = histogram( prediction, 256 )[0]
otsuth = otsu2( hist, maxH * maxW )/256
print(otsuth)
"""
prediction = threshold(prediction, 0.75)

FP, FN, TP, TN, FMEASURE, PWC = numeric_score(prediction, groundtruth)
print('FMEASURE',FMEASURE, 'PWC', PWC)
