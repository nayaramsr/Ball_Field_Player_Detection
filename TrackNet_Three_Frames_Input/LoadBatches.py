import numpy as np
import cv2
import itertools
import csv
import PIL
from PIL import *
from collections import defaultdict


#get input array
def getInputArr( path ,path1 ,path2 , width , height):
	try:
		#img = cv2.imread(path, 1)
		img = PIL.Image.open(path)#np.asarray()

		#read the image
		xi = 256 #2*img.shape[0]/10
		yi = 144 #img.shape[1]/5
		xo = 1024 #8*(output_img.shape[0])/10
		yo = 576 #4*(img.shape[1])/5
		
		cropped_img = img.crop((xi , yi, xo, yo))
		#img = img[y-180:y+180, x-320:x+320]
		
		#resize it 
		img = cv2.resize(np.asarray(cropped_img), ( width , height ))
		
		#input must be float type
		img = img.astype(np.float32)





		#img1 = cv2.imread(path1, 1)
		img1 = PIL.Image.open(path1)
		
		cropped_img1 = img1.crop((xi , yi, xo, yo))
		#img1 = img1[y1-180:y1+180, x1-320:x1+320]

		#resize it 
		img1 = cv2.resize(np.asarray(cropped_img1), ( width , height ))
		
		#input must be float type
		img1 = img1.astype(np.float32)





		#img2 = cv2.imread(path2, 1)
		img2 = PIL.Image.open(path2)
		
		cropped_img2 = img2.crop((xi , yi, xo, yo))
		#img2 = img2[y2-180:y2+180, x2-320:x2+320]

		#resize it 
		img2 = cv2.resize(np.asarray(cropped_img2), ( width , height ))
		
		#input must be float type
		img2 = img2.astype(np.float32)




		#combine three imgs to  (width , height, rgb*3)
		imgs =  np.concatenate((img, img1, img2),axis=2)

		#since the odering of TrackNet  is 'channels_first', so we need to change the axis
		imgs = np.rollaxis(imgs, 2, 0)
		return imgs

	except Exception as e:

		print(path , e)



#get output array
def getOutputArr( anno , nClasses ,  width , height  ):

	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		#img = cv2.imread(anno, 1)
		img = PIL.Image.open(anno)

		xi = 256 #2*img.shape[0]/10
		yi = 144 #img.shape[1]/5
		xo = 1024 #8*(output_img.shape[0])/10
		yo = 576 #4*(img.shape[1])/5
		
		cropped_img = img.crop((xi , yi, xo, yo))
		#img = img[yout-180:yout+180, xout-320:xout+320]

		img = cv2.resize(np.asarray(cropped_img), ( width , height ))
		img = img[:, : , 0]
		#img = img.astype(np.float32)################
		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception as e:
		print(e)
		
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels



#read input data and output data
def InputOutputGenerator( images_path,  batch_size,  n_classes , input_height , input_width , output_height , output_width ):

	#read csv file to 'zipped'
	columns = defaultdict(list)
	with open(images_path) as f:
	    reader = csv.reader(f)
	    next(reader)
	    for row in reader:
	        for (i,v) in enumerate(row):
	            columns[i].append(v)
	zipped = itertools.cycle( zip(columns[0], columns[1], columns[2], columns[3]) )

	while True:
		Input = []
		Output = []
		i = 0

		#read input&output for each batch
		while i < batch_size :
			path, path1, path2 , anno = next(zipped)
			Input.append( getInputArr( path, path1, path2, input_width, input_height) )
			Output.append( getOutputArr( anno , n_classes , output_width , output_height) )
			i += 1

		#return input&output
		yield np.array(Input) , np.array(Output)

