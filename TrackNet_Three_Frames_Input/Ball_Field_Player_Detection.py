import argparse
import Models
import queue
import cv2
import numpy as np
from math import sqrt
import tensorflow as tf
#from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import time
#import opencv_wrapper as cvw
#import sympy
#from sympy.geometry import *
import shapely
from shapely.geometry import LineString, Point
from PIL import Image, ImageDraw

#import imutils

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default = "")
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()
input_video_path =  args.input_video_path
output_video_path =  args.output_video_path
save_weights_path = args.save_weights_path
n_classes =  args.n_classes

if output_video_path == "":
	#output video in same path
	output_video_path = input_video_path.split('.')[0] + "_TrackNet.mp4"

#get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#start from first frame
currentFrame = 0

#width and height in TrackNet
width , height =  640, 360 #320, 240
img, img1, img2 = None, None, None

#load TrackNet model
modelFN = Models.TrackNet.TrackNet
m = modelFN( n_classes , input_height=height, input_width=width   )
m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
m.load_weights(  save_weights_path  )

# In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames 
q = queue.deque()
for i in range(0,15):
	q.appendleft(None)

pontaQuadraBaixoEsquerda = queue.deque()
for i in range(0,10):
	pontaQuadraBaixoEsquerda.appendleft(None)

pontaQuadraBaixoDireita = queue.deque()
for i in range(0,10):
	pontaQuadraBaixoDireita.appendleft(None)

pontaQuadraBaixoEsquerda = queue.deque()
for i in range(0,10):
	pontaQuadraBaixoEsquerda.appendleft(None)

pontaQuadraBaixoDireita = queue.deque()
for i in range(0,10):
	pontaQuadraBaixoDireita.appendleft(None)

#fin = queue.deque()
fin = []

y_0 = np.zeros(15)
x_0 = np.zeros(15)

#deltaAnt = queue.deque()
#for i in range(0,15):
#	deltaAnt.appendleft(None)
deltaAnt = np.ones(15)

deltaY = queue.deque()
for i in range(0,15):
	deltaY.appendleft(None)
deltaY = np.zeros(15)

red_black = np.zeros(15)

#save prediction images as vidoe
#Tutorial: https://stackoverflow.com/questions/33631489/error-during-saving-a-video-using-python-and-opencv
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))


#both first and second frames cant be predict, so we directly write the frames to output video
#capture frame-by-frame
video.set(1,currentFrame); 
ret, img1 = video.read()
#write image to video
output_video.write(img1)
currentFrame +=1
#resize it 
img1 = cv2.resize(img1, ( width , height ))
#input must be float type
img1 = img1.astype(np.float32)

#capture frame-by-frame
video.set(1,currentFrame);
ret, img = video.read()
#write image to video
output_video.write(img)
currentFrame +=1
#resize it 
img = cv2.resize(img, ( width , height ))
#input must be float type
img = img.astype(np.float32)

meio_campo = 540

mais_alto = 540

quicou_cima=False
quicou_baixo=False

yant_cima = meio_campo
yant_baixo = meio_campo

espera_baixo_y = meio_campo
esperamais_baixo_y = meio_campo
yquasebaixo = meio_campo

espera_cima_y= meio_campo
esperamais_cima_y = meio_campo
yquasecima = meio_campo

xo = 0
yo = 0

conta_ai =0 
conta_total = 0

M = 5

color_x=0
color_y=0
#mais alto
#defasar e perder amplitude

#mais baixo
#parece q tem ruido

#n existe filtro q nao defasa

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
lineType               = 2
flag_texto = 0

video.set(1,currentFrame); 
ret, img3 = video.read()
#if there dont have any frame in video, break

output_img =img3

if output_img is not None:

#------------------------Método de detecção de contornos Canny------------------------#

	low_threshold = 100
	high_threshold = 300

	edges = cv2.Canny(output_img, low_threshold, high_threshold)
	cv2.imwrite("edges.png", edges)


	MORPH = 10
	kernel = np.ones( (12,12), np.uint8)


	#------------Filtro para cor laranja do campo para selecionar área de jogo------------#

	lower = np.array([59,83,141], dtype = "uint8")  #60,74,166
	upper = np.array([120,152,206], dtype = "uint8") #[104,163,244] 139,185,255last

	# find the colors within the specified boundaries and apply the mask
	mask = cv2.inRange(output_img, lower, upper)
	cv2.imwrite("mask.png", mask) # - e_im1)

	# Manipulação morfológica da imagem
	d_im = cv2.dilate(edges, kernel, iterations=1)
	cv2.imwrite("d_im.png", d_im)
	e_im1 = cv2.erode(d_im, kernel, iterations=1)
	cv2.imwrite("e_im1.png", e_im1)

	laranja1 = mask - e_im1
	_,laranja1 = cv2.threshold(laranja1,127,255,cv2.THRESH_BINARY)

	laranja1 = cv2.dilate(laranja1, np.ones( (5,5), np.uint8), iterations= 2)
	cv2.imwrite("laranja1.png", laranja1)

	#Contruindo máscara para separação da área de jogo

	contorninho, _ = cv2.findContours( laranja1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

	imprimecontornao = np.copy(output_img) * 0

	for contoorn in contorninho:
		perimeterz = cv2.arcLength(contoorn,True)
		if perimeterz>5000:
			cv2.drawContours(imprimecontornao, [contoorn], 0, (0,255,0), 2)
			cv2.fillPoly(imprimecontornao, pts =[contoorn], color=(255,255,255))
			y_top_imprimecontornao = int(contoorn[contoorn[:,:,1].argmin()][0][1])

	imprimecontornao = cv2.cvtColor(imprimecontornao,cv2.COLOR_BGR2GRAY)

	OUTHL = cv2.bitwise_and(edges,edges, mask = imprimecontornao)


	#-------------------------Detecção de linha usando Hough Lines------------------------#

	lineshoughcinzinha = cv2.HoughLines(OUTHL,1,np.pi/180,250)#cinzinha
	houghlines3 = np.copy(output_img) * 0

	tameita=0
	for eitaa in lineshoughcinzinha:
		for rho,theta in eitaa:

			tameita=tameita+1
	
	linhasfinais = queue.deque()
	for i in range(0,100):
		linhasfinais.appendleft(None)

	for eitaa in lineshoughcinzinha:
		for rho,theta in eitaa:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 3000*(-b))
			y1 = int(y0 + 3000*(a))
			x2 = int(x0 - 3000*(-b))
			y2 = int(y0 - 3000*(a))

			cv2.line(houghlines3,(x1,y1),(x2,y2),(0,255,0),2)

			#cv2.line(output_img,(x1,y1),(x2,y2),(0,0,255),4)

			linhasfinais.appendleft([[x1,y1,x2,y2]])
			#pop x,y from queue
			linhasfinais.pop()





	#-----------------------Detecção de pontos importantes da quadra-----------------------#

	# Pontos visíveis

	inter = queue.deque()
	taminter = 300

	line_image = np.copy(output_img)*0

	for i in range(0,taminter):
		inter.appendleft(None)

	for line1 in linhasfinais:
		for line2 in linhasfinais:
			if line1 is not None and line2 is not None:
				for x1a, y1a, x2a, y2a in line1:
					for x1b, y1b, x2b, y2b in line2:
						if not (x1a==x1b and y1a==y1b and x2a==x2b and y2a==y2b):
							linea = LineString([(x1a, y1a), (x2a, y2a)])
							lineb = LineString([(x1b, y1b), (x2b, y2b)])
							int_pt = linea.intersection(lineb)

							angular_a = (y1a - y2a) / (x1a - x2a)
							linear_a = y1a - (angular_a * x1a)

							angular_b = (y1b - y2b) / (x1b - x2b)
							linear_b = y1b - (angular_b * x1b)

							if (1 + angular_a*angular_b) !=0:

								tangente = abs( (angular_a - angular_b) / (1 + angular_a*angular_b) )
								angulo = abs( np.arctan(tangente) )

							if (1 + angular_a*angular_b) ==0:
								angulo = np.pi/2
							
	
							#point_of_intersection = int_pt.x, int_pt.y
							if np.size(int_pt) !=0 and angulo > np.pi/18 and int(int_pt.coords[0][1]) > 0: #and int(int_pt.coords[0][0])>0 and int(int_pt.coords[0][1])<= 1080 and int(int_pt.coords[0][0])<=1920:	
								cv2.circle(line_image,( int(int_pt.coords[0][0]) , int(int_pt.coords[0][1]) ),5,[0,0,255],-1)
								#push x,y to queue
								inter.appendleft([int(int_pt.coords[0][0]) , int(int_pt.coords[0][1])])
								#pop x,y from queue
								inter.pop()
								#print(q)

	cv2.imwrite("line_image.png", line_image)



	#---

	X_dir_baixo = 0
	X_esq_baixo = 1920

	Y_dir_baixo =0# MENORy
	Y_esq_baixo =1080# MENORy


	for i in range(0,taminter):
		if inter[i] is not None:

			inter_x = inter[i][0]
			inter_y = inter[i][1]

			if inter_x >= X_dir_baixo and inter_x <= 2320 and inter_x > 1920:
				X_dir_baixo = inter_x
				Y_dir_baixo = inter_y

			if inter_x <= X_esq_baixo and inter_x >= -400 and inter_x < 0:
				X_esq_baixo = inter_x
				Y_esq_baixo = inter_y



	#---

	X_dir_baixo_2 = X_esq_baixo
	X_esq_baixo_2 = X_dir_baixo 

	Y_dir_baixo_2 = Y_dir_baixo
	Y_esq_baixo_2 = Y_esq_baixo

	coef_angular = (Y_esq_baixo - Y_dir_baixo) / (X_esq_baixo - X_dir_baixo)
	coef_linear = Y_esq_baixo - (coef_angular * X_esq_baixo)


	for i in range(0,taminter):
		if inter[i] is not None:
				
			inter_x = inter[i][0]
			inter_y = inter[i][1]
			
			if abs( -inter_y + coef_angular * inter_x + coef_linear ) <= 40 and inter_x < X_dir_baixo-50 and inter_x > X_esq_baixo+50 :
				if inter_x >= X_dir_baixo_2:
					X_dir_baixo_2 = inter_x
					Y_dir_baixo_2 = inter_y

				if inter_x <= X_esq_baixo_2:
					X_esq_baixo_2 = inter_x
					Y_esq_baixo_2 = inter_y


	#---

	X_dir_baixo_3 = X_esq_baixo_2 
	X_esq_baixo_3 = X_dir_baixo_2

	Y_dir_baixo_3 = 0
	Y_esq_baixo_3 = 0

	for line1 in linhasfinais:
		for line2 in linhasfinais:
			if line1 is not None and line2 is not None:
				for x1a, y1a, x2a, y2a in line1:
					for x1b, y1b, x2b, y2b in line2:
						if not (x1a==x1b and y1a==y1b and x2a==x2b and y2a==y2b):
							linea = LineString([(x1a, y1a), (x2a, y2a)])
							lineb = LineString([(x1b, y1b), (x2b, y2b)])
							int_pt = linea.intersection(lineb)

							angular_a = (y1a - y2a) / (x1a - x2a)
							linear_a = y1a - (angular_a * x1a)

							angular_b = (y1b - y2b) / (x1b - x2b)
							linear_b = y1b - (angular_b * x1b)
							
							if np.size(int_pt) !=0:

								#point_of_intersection = int_pt.x, int_pt.y
								if int(int_pt.coords[0][0]) < X_esq_baixo_2 + 20 and int(int_pt.coords[0][1]) < Y_esq_baixo_2 + 20 and int(int_pt.coords[0][0]) > X_esq_baixo_2 -20 and int(int_pt.coords[0][1]) > Y_esq_baixo_2 -20:

									if abs( -Y_esq_baixo + angular_a * X_esq_baixo + linear_a ) <= 20:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_b * inter_x + linear_b ) <= 20:

													if inter_y < Y_esq_baixo_2 -40 and inter_y > Y_esq_baixo_3:
														X_esq_baixo_3 = inter_x
														Y_esq_baixo_3 = inter_y

									if abs( -Y_esq_baixo + angular_b * X_esq_baixo + linear_b ) <= 20:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_a * inter_x + linear_a ) <= 20:

													if inter_y < Y_esq_baixo_2 -40 and inter_y > Y_esq_baixo_3:
														X_esq_baixo_3 = inter_x
														Y_esq_baixo_3 = inter_y
		
									

								if int(int_pt.coords[0][0]) < X_dir_baixo_2 + 20 and int(int_pt.coords[0][1]) < Y_dir_baixo_2 + 20 and int(int_pt.coords[0][0]) > X_dir_baixo_2 -20 and int(int_pt.coords[0][1]) > Y_dir_baixo_2 -20:

									if abs(  -Y_esq_baixo + angular_a * X_esq_baixo + linear_a ) <= 20:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_b * inter_x + linear_b ) <= 30:

													if inter_y < Y_dir_baixo_2 -40 and inter_y > Y_dir_baixo_3:
														X_dir_baixo_3 = inter_x
														Y_dir_baixo_3 = inter_y

													

									if abs( -Y_esq_baixo + angular_b * X_esq_baixo + linear_b ) <= 20:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_a * inter_x + linear_a ) <= 40:

													if inter_y < Y_dir_baixo_2 -20 and inter_y > Y_dir_baixo_3:
														X_dir_baixo_3 = inter_x
														Y_dir_baixo_3 = inter_y
	#---

	X_dir_meio =0
	X_esq_meio =0

	Y_dir_meio = 0
	Y_esq_meio = 0

	for line1 in linhasfinais:
		for line2 in linhasfinais:
			if line1 is not None and line2 is not None:
				for x1a, y1a, x2a, y2a in line1:
					for x1b, y1b, x2b, y2b in line2:
						if not (x1a==x1b and y1a==y1b and x2a==x2b and y2a==y2b):
							linea = LineString([(x1a, y1a), (x2a, y2a)])
							lineb = LineString([(x1b, y1b), (x2b, y2b)])
							int_pt = linea.intersection(lineb)

							angular_a = (y1a - y2a) / (x1a - x2a)
							linear_a = y1a - (angular_a * x1a)

							angular_b = (y1b - y2b) / (x1b - x2b)
							linear_b = y1b - (angular_b * x1b)
							
							if np.size(int_pt) !=0:

								if int(int_pt.coords[0][0]) < X_esq_baixo + 20 and int(int_pt.coords[0][1]) < Y_esq_baixo + 20 and int(int_pt.coords[0][0]) > X_esq_baixo - 20 and int(int_pt.coords[0][1]) > Y_esq_baixo - 20:

									if abs( -y1a + coef_angular * x1a + coef_linear ) <= 30:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_b * inter_x + linear_b ) <= 40:

													if inter_y < Y_esq_baixo_3 -60 and inter_y > Y_esq_meio:
														X_esq_meio = inter_x
														Y_esq_meio = inter_y

									if abs( -y1b + coef_angular * x1b + coef_linear ) <= 30:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_a * inter_x + linear_a ) <= 40:

													if inter_y < Y_esq_baixo_3 -60 and inter_y > Y_esq_meio:
														X_esq_meio = inter_x
														Y_esq_meio = inter_y

								if int(int_pt.coords[0][0]) < X_dir_baixo + 20 and int(int_pt.coords[0][1]) < Y_dir_baixo + 20 and int(int_pt.coords[0][0]) > X_dir_baixo - 20 and int(int_pt.coords[0][1]) > Y_dir_baixo - 20:
									
									if abs( -Y_esq_baixo_2 + angular_a * X_esq_baixo_2 + linear_a ) <= 30:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_b * inter_x + linear_b ) <= 30:

													if inter_y < Y_dir_baixo_3 - 60 and inter_y > Y_dir_meio:
														X_dir_meio = inter_x
														Y_dir_meio = inter_y

									if abs( -Y_esq_baixo_2 + angular_b * X_esq_baixo_2 + linear_b ) <= 30:

										for i in range(0,taminter):
											if inter[i] is not None:

														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_a * inter_x + linear_a ) <= 30:

													if inter_y < Y_dir_baixo_3 -100 and inter_y > Y_dir_meio:
														X_dir_meio = inter_x
														Y_dir_meio = inter_y

	#---

	X_baixo_3_meio = 0
	Y_baixo_3_meio = 0

	coef_angular_3 = (Y_esq_baixo_3 - Y_dir_baixo_3) / (X_esq_baixo_3 - X_dir_baixo_3)
	coef_linear_3 = Y_esq_baixo_3 - (coef_angular * X_esq_baixo_3)


	for i in range(0,taminter):
		if inter[i] is not None:
				
			inter_x = inter[i][0]
			inter_y = inter[i][1]
			
			if abs( -inter_y + coef_angular_3 * inter_x + coef_linear_3 ) <= 40 and inter_x < X_dir_baixo_3-50 and inter_x > X_esq_baixo_3+50 :
				X_baixo_3_meio = inter_x
				Y_baixo_3_meio = inter_y


	#---

	X_centro_meio = 0
	Y_centro_meio = 0

	for line1 in linhasfinais:
		for line2 in linhasfinais:
			if line1 is not None and line2 is not None:
				for x1a, y1a, x2a, y2a in line1:
					for x1b, y1b, x2b, y2b in line2:
						if not (x1a==x1b and y1a==y1b and x2a==x2b and y2a==y2b):
							linea = LineString([(x1a, y1a), (x2a, y2a)])
							lineb = LineString([(x1b, y1b), (x2b, y2b)])
							int_pt = linea.intersection(lineb)

							angular_a = (y1a - y2a) / (x1a - x2a)
							linear_a = y1a - (angular_a * x1a)

							angular_b = (y1b - y2b) / (x1b - x2b)
							linear_b = y1b - (angular_b * x1b)
							
							if np.size(int_pt) !=0:
								#point_of_intersection = int_pt.x, int_pt.y
								if int(int_pt.coords[0][0]) == X_baixo_3_meio and int(int_pt.coords[0][1]) == Y_baixo_3_meio:

									if abs( -y1a + coef_angular_3 * x1a + coef_linear_3 ) <= 30:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_b * inter_x + linear_b ) <= 40:

													if inter_y < Y_baixo_3_meio -20 and inter_y > Y_centro_meio:
														X_centro_meio = inter_x
														Y_centro_meio = inter_y

									if abs( -y1b + coef_angular_3 * x1b + coef_linear_3 ) <= 30:

										for i in range(0,taminter):
											if inter[i] is not None:
														
												inter_x = inter[i][0]
												inter_y = inter[i][1]
													
												if abs( -inter_y + angular_a * inter_x + linear_a ) <= 40:

													if inter_y < Y_baixo_3_meio -20 and inter_y > Y_centro_meio:
														X_centro_meio = inter_x
														Y_centro_meio = inter_y
			
	# Média dos pontos

	soma_x_dir_baixo = 0
	soma_y_dir_baixo = 0
	n_dir_baixo = 0
	
	soma_x_esq_baixo = 0
	soma_y_esq_baixo = 0
	n_esq_baixo = 0



	soma_x_dir_baixo_2 = 0
	soma_y_dir_baixo_2 = 0
	n_dir_baixo_2 = 0

	soma_x_esq_baixo_2 = 0
	soma_y_esq_baixo_2 = 0
	n_esq_baixo_2 = 0



	soma_x_dir_baixo_3 = 0
	soma_y_dir_baixo_3 = 0
	n_dir_baixo_3 = 0

	soma_x_esq_baixo_3 = 0
	soma_y_esq_baixo_3 = 0
	n_esq_baixo_3 = 0

	soma_x_baixo_3_meio = 0
	soma_y_baixo_3_meio = 0
	n_baixo_3_meio = 0



	soma_x_dir_meio = 0
	soma_y_dir_meio = 0
	n_dir_meio = 0

	soma_x_esq_meio = 0
	soma_y_esq_meio = 0
	n_esq_meio = 0

	soma_x_centro_meio = 0
	soma_y_centro_meio = 0
	n_centro_meio = 0


	for i in range(0,taminter):
		if inter[i] is not None:

			inter_x = inter[i][0]
			inter_y = inter[i][1]

			if sqrt( (inter_x-X_dir_baixo)**2 + (inter_y-Y_dir_baixo)**2 )<100 or (inter_x == X_dir_baixo and inter_y == Y_dir_baixo):
				soma_x_dir_baixo = soma_x_dir_baixo + inter_x
				soma_y_dir_baixo = soma_y_dir_baixo + inter_y
				n_dir_baixo = n_dir_baixo +1

			if sqrt( (inter_x-X_esq_baixo)**2 + (inter_y-Y_esq_baixo)**2 )<100 or (inter_x == X_esq_baixo and inter_y == Y_esq_baixo):
				soma_x_esq_baixo = soma_x_esq_baixo + inter_x
				soma_y_esq_baixo = soma_y_esq_baixo + inter_y
				n_esq_baixo = n_esq_baixo +1

		#---

			inter_x_2 = inter[i][0]
			inter_y_2 = inter[i][1]

			if sqrt( (inter_x_2-X_dir_baixo_2)**2 + (inter_y_2-Y_dir_baixo_2)**2 )<100 or (inter_x_2 == X_dir_baixo_2 and inter_y_2 == Y_dir_baixo_2):
				soma_x_dir_baixo_2 = soma_x_dir_baixo_2 + inter_x_2
				soma_y_dir_baixo_2 = soma_y_dir_baixo_2 + inter_y_2
				n_dir_baixo_2 = n_dir_baixo_2 +1

			if sqrt( (inter_x_2-X_esq_baixo_2)**2 + (inter_y_2-Y_esq_baixo_2)**2 )<100 or (inter_x_2 == X_esq_baixo_2 and inter_y_2 == Y_esq_baixo_2):
				soma_x_esq_baixo_2 = soma_x_esq_baixo_2 + inter_x_2
				soma_y_esq_baixo_2 = soma_y_esq_baixo_2 + inter_y_2
				n_esq_baixo_2 = n_esq_baixo_2 +1


		#---

			inter_x_3 = inter[i][0]
			inter_y_3 = inter[i][1]

			if sqrt( (inter_x_3-X_dir_baixo_3)**2 + (inter_y_3-Y_dir_baixo_3)**2 )<100 or (inter_x_3 == X_dir_baixo_3 and inter_y_3 == Y_dir_baixo_3):
				soma_x_dir_baixo_3 = soma_x_dir_baixo_3 + inter_x_3
				soma_y_dir_baixo_3 = soma_y_dir_baixo_3 + inter_y_3
				n_dir_baixo_3 = n_dir_baixo_3 +1

			if sqrt( (inter_x_3-X_esq_baixo_3)**2 + (inter_y_3-Y_esq_baixo_3)**2 )<100 or (inter_x_3 == X_esq_baixo_3 and inter_y_3 == Y_esq_baixo_3):
				soma_x_esq_baixo_3 = soma_x_esq_baixo_3 + inter_x_3
				soma_y_esq_baixo_3 = soma_y_esq_baixo_3 + inter_y_3
				n_esq_baixo_3 = n_esq_baixo_3 +1


		#---
			inter_x_baixo_3_meio = inter[i][0]
			inter_y_baixo_3_meio = inter[i][1]

			if sqrt( (inter_x_baixo_3_meio-X_baixo_3_meio)**2 + (inter_y_baixo_3_meio-Y_baixo_3_meio)**2 )<100 or (inter_x_baixo_3_meio == X_baixo_3_meio and inter_y_baixo_3_meio == Y_baixo_3_meio):
				soma_x_baixo_3_meio = soma_x_baixo_3_meio + inter_x_baixo_3_meio
				soma_y_baixo_3_meio = soma_y_baixo_3_meio + inter_y_baixo_3_meio
				n_baixo_3_meio = n_baixo_3_meio +1
	
		#---

			inter_x_meio = inter[i][0]
			inter_y_meio = inter[i][1]

			if sqrt( (inter_x_meio-X_dir_meio)**2 + (inter_y_meio-Y_dir_meio)**2 )<60 or (inter_x_meio == X_dir_meio and inter_y_meio == Y_dir_meio):
				soma_x_dir_meio = soma_x_dir_meio + inter_x_meio
				soma_y_dir_meio = soma_y_dir_meio + inter_y_meio
				n_dir_meio = n_dir_meio +1

			if sqrt( (inter_x_meio-X_esq_meio)**2 + (inter_y_meio-Y_esq_meio)**2 )<60 or (inter_x_meio == X_esq_meio and inter_y_meio == Y_esq_meio):
				soma_x_esq_meio = soma_x_esq_meio + inter_x_meio
				soma_y_esq_meio = soma_y_esq_meio + inter_y_meio
				n_esq_meio = n_esq_meio +1


		#---

			inter_x_centro_meio = inter[i][0]
			inter_y_centro_meio = inter[i][1]

			if sqrt( (inter_x_centro_meio-X_centro_meio)**2 + (inter_y_centro_meio-Y_centro_meio)**2 )<60 or (inter_x_centro_meio == X_centro_meio and inter_y_centro_meio == Y_centro_meio):
				soma_x_centro_meio = soma_x_centro_meio + inter_x_centro_meio
				soma_y_centro_meio = soma_y_centro_meio + inter_y_centro_meio
				n_centro_meio = n_centro_meio +1



	if n_dir_baixo !=0:
		X_dir_baixo = int(soma_x_dir_baixo / n_dir_baixo)
	if n_esq_baixo !=0:
		X_esq_baixo = int(soma_x_esq_baixo / n_esq_baixo)

	if n_dir_baixo !=0:
		Y_dir_baixo = int(soma_y_dir_baixo / n_dir_baixo)
	if n_esq_baixo !=0:
		Y_esq_baixo = int(soma_y_esq_baixo / n_esq_baixo)



	if n_dir_baixo_2 !=0:
		X_dir_baixo_2 = int(soma_x_dir_baixo_2 / n_dir_baixo_2)
	if n_esq_baixo_2 !=0:
		X_esq_baixo_2 = int(soma_x_esq_baixo_2 / n_esq_baixo_2)

	if n_dir_baixo_2 !=0:
		Y_dir_baixo_2 = int(soma_y_dir_baixo_2 / n_dir_baixo_2)
	if n_esq_baixo_2 !=0:
		Y_esq_baixo_2 = int(soma_y_esq_baixo_2 / n_esq_baixo_2)



	if n_dir_baixo_3 !=0:
		X_dir_baixo_3 = int(soma_x_dir_baixo_3 / n_dir_baixo_3)
	if n_esq_baixo_3 !=0:
		X_esq_baixo_3 = int(soma_x_esq_baixo_3 / n_esq_baixo_3)

	if n_dir_baixo_3 !=0:
		Y_dir_baixo_3 = int(soma_y_dir_baixo_3 / n_dir_baixo_3)
	if n_esq_baixo_3 !=0:
		Y_esq_baixo_3 = int(soma_y_esq_baixo_3 / n_esq_baixo_3)



	if n_dir_meio !=0:
		X_baixo_3_meio = int(soma_x_baixo_3_meio / n_baixo_3_meio)

	if n_esq_meio !=0:
		Y_baixo_3_meio = int(soma_y_baixo_3_meio / n_baixo_3_meio)



	if n_dir_meio !=0:
		X_dir_meio = int(soma_x_dir_meio / n_dir_meio)
	if n_esq_meio !=0:
		X_esq_meio = int(soma_x_esq_meio / n_esq_meio)

	if n_dir_meio !=0:
		Y_dir_meio = int(soma_y_dir_meio / n_dir_meio)
	if n_esq_meio !=0:
		Y_esq_meio = int(soma_y_esq_meio / n_esq_meio)



	if n_centro_meio !=0:
		X_centro_meio = int(soma_x_centro_meio / n_centro_meio)

	if n_centro_meio !=0:
		Y_centro_meio = int(soma_y_centro_meio / n_centro_meio)


	cv2.circle(output_img,( X_dir_baixo , Y_dir_baixo ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_esq_baixo , Y_esq_baixo ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_dir_baixo_2 , Y_dir_baixo_2 ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_esq_baixo_2 , Y_esq_baixo_2 ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_dir_baixo_3 , Y_dir_baixo_3 ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_esq_baixo_3 , Y_esq_baixo_3 ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_baixo_3_meio , Y_baixo_3_meio ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_dir_meio , Y_dir_meio ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_esq_meio , Y_esq_meio ),10,[0,0,255],-1)
	cv2.circle(output_img,( X_centro_meio , Y_centro_meio ),10,[0,0,255],-1)


	# Pontos estimados

	Cx = X_centro_meio + (X_centro_meio - X_dir_baixo) /2;
	Cy = Y_centro_meio + (Y_centro_meio - Y_dir_baixo) / 2;

	cruzad1 = LineString([(int(Cx),int(Cy)), (X_dir_baixo,Y_dir_baixo)])

	#cv2.line(output_img,(int(Cx),int(Cy)), (X_dir_baixo,Y_dir_baixo),(0,255,0),2)

	Cx = X_esq_meio + (X_esq_meio - X_esq_baixo) / 2;
	Cy = Y_esq_meio + (Y_esq_meio - Y_esq_baixo) / 2;

	cruzad2 = LineString([(X_esq_baixo,Y_esq_baixo), (int(Cx),int(Cy))])

	#cv2.line(output_img,(X_esq_baixo,Y_esq_baixo), (int(Cx),int(Cy)),(0,255,0),2)

	esq_alto = cruzad1.intersection(cruzad2)
	X_esq_alto = int(esq_alto.coords[0][0])
	Y_esq_alto = int(esq_alto.coords[0][1])

	cv2.circle(output_img,( X_esq_alto , Y_esq_alto ),3,[0,255,0],-1)






	Cx = X_centro_meio + (X_centro_meio - X_dir_baixo_2) / 2;
	Cy = Y_centro_meio + (Y_centro_meio - Y_dir_baixo_2) / 2;

	cruzad3 = LineString([(int(Cx),int(Cy)), (X_dir_baixo_2,Y_dir_baixo_2)])

	Cx = X_esq_baixo_3 + (X_esq_baixo_3 - X_esq_baixo_2) / 1;
	Cy = Y_esq_baixo_3 + (Y_esq_baixo_3 - Y_esq_baixo_2) / 1;

	cruzad4 = LineString([(X_esq_baixo_2,Y_esq_baixo_2), (int(Cx),int(Cy))])

	esq_alto_2 = cruzad3.intersection(cruzad4)
	X_esq_alto_2 = int(esq_alto_2.coords[0][0])
	Y_esq_alto_2 = int(esq_alto_2.coords[0][1])

	cv2.circle(output_img,( X_esq_alto_2 , Y_esq_alto_2 ),3,[0,255,0],-1)






	Cx = X_centro_meio + (X_centro_meio - X_esq_baixo) / 2;
	Cy = Y_centro_meio + (Y_centro_meio - Y_esq_baixo) / 2;

	cruzad5 = LineString([(int(Cx),int(Cy)), (X_esq_baixo,Y_esq_baixo)])

	Cx = X_dir_meio + (X_dir_meio - X_dir_baixo) / 1;
	Cy = Y_dir_meio + (Y_dir_meio - Y_dir_baixo) / 1;

	cruzad6 = LineString([(X_dir_baixo,Y_dir_baixo), (int(Cx),int(Cy))])

	dir_alto = cruzad5.intersection(cruzad6)
	X_dir_alto = int(dir_alto.coords[0][0])
	Y_dir_alto = int(dir_alto.coords[0][1])

	cv2.circle(output_img,( X_dir_alto , Y_dir_alto ),3,[0,255,0],-1)






	Cx = X_centro_meio + (X_centro_meio - X_esq_baixo_2) / 2;
	Cy = Y_centro_meio + (Y_centro_meio - Y_esq_baixo_2) / 2;

	cruzad7 = LineString([(int(Cx),int(Cy)), (X_esq_baixo_2,Y_esq_baixo_2)])

	Cx = X_dir_baixo_3 + (X_dir_baixo_3 - X_dir_baixo_2) / 1;
	Cy = Y_dir_baixo_3 + (Y_dir_baixo_3 - Y_dir_baixo_2) / 1;

	cruzad8 = LineString([(X_dir_baixo_2,Y_dir_baixo_2), (int(Cx),int(Cy))])

	dir_alto_2 = cruzad7.intersection(cruzad8)
	X_dir_alto_2 = int(dir_alto_2.coords[0][0])
	Y_dir_alto_2 = int(dir_alto_2.coords[0][1])

	cv2.circle(output_img,( X_dir_alto_2 , Y_dir_alto_2 ),3,[0,255,0],-1)






	

	Cx = X_centro_meio + (X_centro_meio - X_dir_baixo_3) / 2;
	Cy = Y_centro_meio + (Y_centro_meio - Y_dir_baixo_3) / 2;

	cruzad9 = LineString([(int(Cx),int(Cy)), (X_dir_baixo_3,Y_dir_baixo_3)])

	Cx = X_esq_baixo_3 + (X_esq_baixo_3 - X_esq_baixo_2) / 1;
	Cy = Y_esq_baixo_3 + (Y_esq_baixo_3 - Y_esq_baixo_2) / 1;

	cruzad10 = LineString([(X_esq_baixo_2,Y_esq_baixo_2), (int(Cx),int(Cy))])

	esq_alto_3 = cruzad9.intersection(cruzad10)
	X_esq_alto_3 = int(esq_alto_3.coords[0][0])
	Y_esq_alto_3 = int(esq_alto_3.coords[0][1])

	cv2.circle(output_img,( X_esq_alto_3 , Y_esq_alto_3 ),3,[0,255,0],-1)






	Cx = X_centro_meio + (X_centro_meio - X_esq_baixo_3) / 1;
	Cy = Y_centro_meio + (Y_centro_meio - Y_esq_baixo_3) / 1;

	cruzad11 = LineString([(int(Cx),int(Cy)), (X_esq_baixo_3,Y_esq_baixo_3)])

	Cx = X_dir_baixo_3 + (X_dir_baixo_3 - X_dir_baixo_2) / 1;
	Cy = Y_dir_baixo_3 + (Y_dir_baixo_3 - Y_dir_baixo_2) / 1;

	cruzad12 = LineString([(X_dir_baixo_2,Y_dir_baixo_2), (int(Cx),int(Cy))])

	dir_alto_3 = cruzad11.intersection(cruzad12)
	X_dir_alto_3 = int(dir_alto_3.coords[0][0])
	Y_dir_alto_3 = int(dir_alto_3.coords[0][1])

	cv2.circle(output_img,( X_dir_alto_3 , Y_dir_alto_3 ),3,[0,255,0],-1)






	Cx = X_centro_meio + (X_centro_meio - X_baixo_3_meio) / 1;
	Cy = Y_centro_meio + (Y_centro_meio - Y_baixo_3_meio) / 1;

	cruzad13 = LineString([(int(Cx),int(Cy)), (X_baixo_3_meio,Y_baixo_3_meio)])

	cruzad14 = LineString([(X_dir_alto_3,Y_dir_alto_3), (X_esq_alto_3,Y_esq_alto_3)])

	alto_3_meio = cruzad13.intersection(cruzad14)
	X_alto_3_meio = int(alto_3_meio.coords[0][0])
	Y_alto_3_meio = int(alto_3_meio.coords[0][1])

	cv2.circle(output_img,( X_alto_3_meio , Y_alto_3_meio ),3,[0,255,0],-1)






#	cv2.line(output_img,(X_dir_baixo, Y_dir_baixo),(X_esq_baixo, Y_esq_baixo),(0,255,0),2)
#	cv2.line(output_img,(X_dir_baixo, Y_dir_baixo),(X_dir_alto, Y_dir_alto),(0,255,0),2)
#	cv2.line(output_img,(X_esq_baixo, Y_esq_baixo),(X_esq_alto, Y_esq_alto),(0,255,0),2)
#	cv2.line(output_img,(X_dir_alto, Y_dir_alto),(X_esq_alto, Y_esq_alto),(0,255,0),2)

#	cv2.line(output_img,(X_dir_baixo_2, Y_dir_baixo_2),(X_dir_alto_2, Y_dir_alto_2),(0,255,0),2)
#	cv2.line(output_img,(X_esq_baixo_2, Y_esq_baixo_2),(X_esq_alto_2, Y_esq_alto_2),(0,255,0),2)

#	cv2.line(output_img,(X_dir_baixo_3, Y_dir_baixo_3),(X_esq_baixo_3, Y_esq_baixo_3),(0,255,0),2)
#	cv2.line(output_img,(X_dir_alto_3, Y_dir_alto_3),(X_esq_alto_3, Y_esq_alto_3),(0,255,0),2)

#	cv2.line(output_img,( X_dir_meio , Y_dir_meio ),( X_esq_meio , Y_esq_meio ),(0,255,0),2)

#	cv2.line(output_img,( X_alto_3_meio , Y_alto_3_meio ),( X_baixo_3_meio , Y_baixo_3_meio ),(0,255,0),4)

#	cv2.imwrite("output_img_por_enquanto.png", output_img)
	
	

while(True): #and currentFrame<150): # PODE USAR O CODIGO SEM CORTAR O VIDEO

	img2 = img1
	img1 = img

	linhasfinais = queue.deque()
	for i in range(0,100):
		linhasfinais.appendleft(None)

	#capture frame-by-frame
	video.set(1,currentFrame); 
	ret, img = video.read()

	#if there dont have any frame in video, break
	if not ret: 
		break

	#img is the frame that TrackNet will predict the position
	#since we need to change the size and type of img, copy it to output_img
	output_img = img

	#resize it 
	img = cv2.resize(img, ( width , height ))
	#input must be float type
	img = img.astype(np.float32)


	#combine three imgs to  (width , height, rgb*3)
	X =  np.concatenate((img, img1, img2),axis=2)

	#since the odering of TrackNet  is 'channels_first', so we need to change the axis
	X = np.rollaxis(X, 2, 0)
	#prdict heatmap
	pr = m.predict( np.array([X]) )[0]

	#since TrackNet output is ( net_output_height*model_output_width , n_classes )
	#so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
	#.argmax( axis=2 ) => select the largest probability as class
	pr = pr.reshape(( height ,  width , n_classes ) ).argmax( axis=2 )

	#cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
	pr = pr.astype(np.uint8) 

	#reshape the image size as original input image
	heatmap = cv2.resize(pr  , (output_width, output_height ))

	#heatmap is converted into a binary image by threshold method.
	ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)

	#find the circle in image with 2<=radius<=7
	circles = cv2.HoughCircles(heatmap,cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2, maxRadius=30)



	cinziz = np.copy(output_img) * 0
#	manodosky = np.copy(houghcinzinha)*0

#	cv2.line(manodosky,(X_dir_baixo, Y_dir_baixo),(X_esq_baixo, Y_esq_baixo),(255,255,255),1)
#	cv2.line(manodosky,(X_dir_baixo, Y_dir_baixo),(X_dir_alto, Y_dir_alto),(255,255,255),1)
#	cv2.line(manodosky,(X_esq_baixo, Y_esq_baixo),(X_esq_alto, Y_esq_alto),(255,255,255),1)
#	cv2.line(manodosky,(X_dir_alto, Y_dir_alto),(X_esq_alto, Y_esq_alto),(255,255,255),1)
#	cv2.line(manodosky,(1920,0),(1920,1080),(255,255,255),4)
#	cv2.line(manodosky,(0,0),(0,1080),(255,255,255),4)

#	cv2.imwrite("manodosky.png", manodosky)

#	_,contorno_quadra,_ = cv2.findContours( manodosky, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
#	cv2.fillPoly(manodosky, contorno_quadra, color=(255,255,255))

#	cv2.imwrite("manodosky.png", manodosky) 


############################################################################################################


	manodosky = np.copy(output_img) * 0
	manodosky1 = np.copy(output_img) * 0

	contorno_quadra = np.array([[X_dir_alto,Y_dir_alto],[X_esq_alto,Y_esq_alto],[X_dir_baixo,Y_dir_baixo],[X_esq_baixo,Y_esq_baixo]])

	cv2.line(manodosky,(X_dir_baixo, Y_dir_baixo),(X_esq_baixo, Y_esq_baixo),(255,255,255),1)
	cv2.line(manodosky,(X_dir_baixo, Y_dir_baixo),(X_dir_alto, Y_dir_alto),(255,255,255),1)
	cv2.line(manodosky,(X_esq_baixo, Y_esq_baixo),(X_esq_alto, Y_esq_alto),(255,255,255),1)
	cv2.line(manodosky,(X_dir_alto, Y_dir_alto),(X_esq_alto, Y_esq_alto),(255,255,255),1)
	cv2.line(manodosky,(1920,0),(1920,1080),(255,255,255),4)
	cv2.line(manodosky,(0,0),(0,1080),(255,255,255),4)	

	cv2.line(manodosky1,(X_dir_baixo+100, Y_dir_baixo+100),(X_esq_baixo-100, Y_esq_baixo+100),(255,255,255),1)
	cv2.line(manodosky1,(X_dir_baixo+100, Y_dir_baixo+100),(X_dir_alto+50, Y_dir_alto-50),(255,255,255),1)
	cv2.line(manodosky1,(X_esq_baixo-100, Y_esq_baixo+100),(X_esq_alto-50, Y_esq_alto-50),(255,255,255),1)
	cv2.line(manodosky1,(X_dir_alto+50, Y_dir_alto-50),(X_esq_alto-50, Y_esq_alto-50),(255,255,255),1)
	cv2.line(manodosky1,(1920,0),(1920,1080),(255,255,255),4)
	cv2.line(manodosky1,(0,0),(0,1080),(255,255,255),4)	
	cv2.line(manodosky1,(0,1080),(1920,1080),(255,255,255),4)

	contorno_quadra,_ = cv2.findContours( cv2.cvtColor(manodosky, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )

	contorno_quadra_grande_area,_ = cv2.findContours( cv2.cvtColor(manodosky1, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )


	for contQuadra in contorno_quadra_grande_area:
		cv2.fillPoly(manodosky1, pts =[contQuadra], color=(255,255,255))

	manodosky1 = cv2.cvtColor(manodosky1,cv2.COLOR_BGR2GRAY)
		

	saida_com_erosao = cv2.erode(output_img, np.ones( (12,12), np.uint8) , iterations=1)
	grad = cv2.morphologyEx( saida_com_erosao, cv2.MORPH_GRADIENT, np.ones( (8,8), np.uint8) )
	gradi = cv2.morphologyEx( grad, cv2.MORPH_GRADIENT, np.ones( (10,10), np.uint8) )

	gradie = cv2.bitwise_and(grad, gradi, mask = manodosky1)

	oque = cv2.bitwise_and(gradie,gradie, mask = manodosky1)


	gradient = cv2.Canny(oque, low_threshold, high_threshold)

	cv2.imwrite("saida_com_erosao.png", saida_com_erosao)
	cv2.imwrite("manodosky1.png", manodosky1)
	cv2.imwrite("grad.png", grad)
	cv2.imwrite("gradi.png", gradi)
	cv2.imwrite("gradie.png", gradie)

	cv2.imwrite("oque.png", oque)

	quase_pessoas = np.copy(cinziz)*0
	quase_pessoas_cinza = cv2.cvtColor(quase_pessoas,cv2.COLOR_BGR2GRAY)

	gradient = cv2.dilate(gradient, np.ones( (9,9), np.uint8) ,iterations = 2)
	cont_gradient, _ = cv2.findContours( gradient, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE ) #LIST

	cv2.imwrite("gradient.png", gradient)

	jog_x = 0
	jog_y = 0
	jog_w = 0
	jog_h = 0

	maior_area_final = 400
	for grand_gradient in cont_gradient:
		#cv2.drawContours(gradient, [grand_gradient], 0, (255,255,255), 2)
		
		xxx,yyy,www,hhh = cv2.boundingRect(grand_gradient)
		#print(xxx,yyy,www,hhh)
		AREAcinziz = www*hhh
		#print(AREAcinziz)

		if AREAcinziz > maior_area_final and yyy+hhh > 386 and AREAcinziz > 20000: #and AREAcinziz < maior_area_final :###########################################
			maior_area_final = AREAcinziz
			jog_x = xxx
			jog_y = yyy
			jog_w = www
			jog_h = hhh

	#print("maior_area_final")
	#print(maior_area_final)

	output_img = cv2.rectangle(output_img,(jog_x,jog_y),(jog_x+jog_w,jog_y+jog_h),(255,0,0),3)
############################################################################################################

	#for gradient_fim_uhuuul in cont_gradient:

		#xxx,yyy,www,hhh = cv2.boundingRect(grand_gradient)
		#AREAcinziz = www*hhh

		#if AREAcinziz == maior_area_final:
		#	jog_x,jog_y,jog_w,jog_h = cv2.boundingRect(grand_gradient)
			


		#cinziz = cv2.rectangle(cinziz,(xxx,yyy),(xxx+www,yyy+hhh),(0,255,0),2)
		#quase_pessoas = cv2.rectangle(quase_pessoas,(xxx,yyy),(xxx+www,yyy+hhh),(0,255,0),2)

	cont_quase_pessoas, _ = cv2.findContours( quase_pessoas_cinza, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

	for grand_quase_pessoas in cont_quase_pessoas:
		#verif1= cv2.pointPolygonTest(grand_quase_pessoas, (int(pt_centro_dir.coords[0][0]),int(pt_centro_dir.coords[0][1])) , False)
		#verif2= cv2.pointPolygonTest(grand_quase_pessoas, (int(pt_centro_esq.coords[0][0]),int(pt_centro_esq.coords[0][1])) , False)
		if (cv2.contourArea(grand_quase_pessoas) > 60) :
			#if verif1<0 and verif2<0:
				xxxx,yyyy,wwww,hhhh = cv2.boundingRect(grand_quase_pessoas)
				quase_pessoas = cv2.rectangle(quase_pessoas,(xxxx+10,yyyy+10),(xxxx+wwww,yyyy+hhhh),(0,0,255),3)

	cv2.imwrite("quase_pessoas.png", quase_pessoas)
	cv2.imwrite("cinziz.png", cinziz)
	cv2.imwrite("gradient.png", gradient)
	

	#cv2.imshow('houghlines.png', line_image)
	#cv2.imwrite("line_image.png", line_image)  
	houghimage = cv2.imread("line_image.png") 

	gray = cv2.cvtColor(houghimage,cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray,11, 17, 17)

	

	#cv2.imshow("Lines4",closing)
	ccc = -1
	achei = 0
	contours, h = cv2.findContours( gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
	for cont in contours:
		if achei!=1:
			ccc = ccc +1
		verif= cv2.pointPolygonTest(cont, (660,470), False)
		if verif>=0 :
			achei = 1
			approx = cv2.approxPolyDP(cont, 0.001*cv2.arcLength(cont,True), True)
	#		cv2.drawContours(output_img, [approx], 0, (0), 2)

	cv2.imwrite("output_img_saibro_ponta.png", output_img)




	#In order to draw the circle in output_img, we need to used PIL library
	#Convert opencv image format to PIL image format
	PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB) 
	#PIL_image =  255*np.ones([output_img.shape[0], output_img.shape[1],3],'uint8')
	PIL_image = Image.fromarray(PIL_image)


	cv2.line(output_img,(X_dir_baixo, Y_dir_baixo),(X_esq_baixo, Y_esq_baixo),(0,255,0),2)
	cv2.line(output_img,(X_dir_baixo, Y_dir_baixo),(X_dir_alto, Y_dir_alto),(0,255,0),2)
	cv2.line(output_img,(X_esq_baixo, Y_esq_baixo),(X_esq_alto, Y_esq_alto),(0,255,0),2)
	cv2.line(output_img,(X_dir_alto, Y_dir_alto),(X_esq_alto, Y_esq_alto),(0,255,0),2)

	cv2.line(output_img,(X_dir_baixo_2, Y_dir_baixo_2),(X_dir_alto_2, Y_dir_alto_2),(0,255,0),2)
	cv2.line(output_img,(X_esq_baixo_2, Y_esq_baixo_2),(X_esq_alto_2, Y_esq_alto_2),(0,255,0),2)

	cv2.line(output_img,(X_dir_baixo_3, Y_dir_baixo_3),(X_esq_baixo_3, Y_esq_baixo_3),(0,255,0),2)
	cv2.line(output_img,(X_dir_alto_3, Y_dir_alto_3),(X_esq_alto_3, Y_esq_alto_3),(0,255,0),2)

	cv2.line(output_img,( X_dir_meio , Y_dir_meio ),( X_esq_meio , Y_esq_meio ),(0,255,0),2)

	cv2.line(output_img,( X_alto_3_meio , Y_alto_3_meio ),( X_baixo_3_meio , Y_baixo_3_meio ),(0,255,0),4)



	#check if there have any tennis be detected
	if circles is not None:
		#if only one tennis be detected
		conta_total += 1
		if currentFrame == 0:
			xo = int(circles[0][0][0])
			yo = int(circles[0][0][1])
				
			#if yo > (output_img.shape[1] /7):
			#push x,y to queue
			q.appendleft([xo,yo])   
			#pop x,y from queue
			q.pop()
			print(currentFrame, xo,yo)
			#print(q)

			#bbox =  (xo - 2, xo - 2, xo + 2, xo + 2)
			#draw = ImageDraw.Draw(PIL_image)
			#draw.ellipse(bbox, outline ='red')
	
		else:
			x = int(circles[0][0][0])
			y = int(circles[0][0][1]) 

			dif = sqrt( (x-xo)**2 + (y-yo)**2 )

			if dif>= 5 and dif <= 40: #76:
				#if y > (output_img.shape[1] /7):

					#if x > (output_img.shape[0]/10) and x < (9*(output_img.shape[0])/10):
	
						print(currentFrame, x, y, xo, yo, dif)

						#push x,y to queue
						q.appendleft([x,y])
						#pop x,y from queue
						q.pop()
						#print(q)

			else:
				if len(circles[0]) > 1:

					x = int(circles[0][1][0])
					y = int(circles[0][1][1]) 

					dif = sqrt( (x-xo)**2 + (y-yo)**2 )

					if dif>= 5 and dif <= 40: #76:
						#if y > (output_img.shape[1] /7):

							#if x > (output_img.shape[0]/10) and x < (9*(output_img.shape[0])/10):
	
								print(currentFrame, x, y, xo, yo, dif)

								#push x,y to queue
								q.appendleft([x,y])
								#pop x,y from queue
								q.pop()
								#print(q)


					else:
						if len(circles[0]) > 2:
							x = int(circles[0][2][0])
							y = int(circles[0][2][1])

							dif = sqrt( (x-xo)**2 + (y-yo)**2 )

							if dif>= 5 and dif <= 40: #76:
								#if y > (output_img.shape[1] /7):

									#if x > (output_img.shape[0]/10) and x < (9*(output_img.shape[0])/10):
	
										print(currentFrame, x, y, xo, yo, dif)


										#push x,y to queue
										q.appendleft([x,y])
										#pop x,y from queue
										q.pop()
										#print(q)

										
							else:
								if len(circles[0]) > 3:
									x = int(circles[0][3][0])
									y = int(circles[0][3][1])

									dif = sqrt( (x-xo)**2 + (y-yo)**2 )

									if dif>= 5 and dif <= 40: #76:
										#if y > (output_img.shape[1] /7):

											#if x > (output_img.shape[0]/10) and x < (9*(output_img.shape[0])/10):
	
												print(currentFrame, x, y, xo, yo, dif)

												#push x,y to queue
												q.appendleft([x,y])
												#pop x,y from queue
												q.pop()
												#print(q)
										
									else:
										conta_ai += 1
										#print(currentFrame, x, y, xo, yo, dif)

			xo = x
			yo = y


	#next frame
	currentFrame += 1
	

	#print(q)
	if len(q) < 15:
		tam = len(q)
	else:
		tam = 15
	
	
	if q[14] is not None:

		for i in range(0,tam):
			if q[i] is not None:
				
				draw_x = q[i][0]
				draw_y = q[i][1]
	
				deltaY[i] = draw_y - y_0[i]

				if i==0:
					red_black[i]= 0 #yellow
					
				else:
					if i>3:
						if ( deltaAnt[i-2] is not None ) and ( deltaAnt[i-1] is not None ) and ( deltaAnt[i] is not None ) :
	
							if (deltaAnt[i-2] > 0) and (deltaAnt[i-1] > 0) and (deltaAnt[i] < 0) and (deltaY[i] < 0):
								red_black[i-2]= 1 #red
	
							else:
								red_black[i-2]= 2 #black
					else:	
						if i>0 : 	
							red_black[i]= 2	#black	
	
				if i < tam-1 and i >= 1:
					y_0[i+1] = draw_y
					x_0[i+1] = draw_x
					deltaAnt[i+1] = deltaY[i]

		for i in range(0,len(red_black)-2):
			if q[i] is not None:
				color_x = q[i][0]
				color_y = q[i][1]

				if red_black[i] == 0:
					bbox =  (color_x - 2, color_y - 2, color_x + 2, color_y + 2)
					draw = ImageDraw.Draw(PIL_image)
					draw.ellipse(bbox, outline ='yellow')
					del draw
	
				if red_black[i] == 1:
					bbox =  (color_x - 2, color_y - 2, color_x + 2, color_y + 2)		
					draw = ImageDraw.Draw(PIL_image)
					draw.ellipse(bbox, outline ='red')
					flag_texto = 5
					x_quique = color_x
					y_quique = color_y
					del draw
	
				if red_black[i] == 2:
					bbox =  (color_x - 2, color_y - 2, color_x + 2, color_y + 2)		
					draw = ImageDraw.Draw(PIL_image)
					draw.ellipse(bbox, outline ='black')
					del draw

	

	cv2.imwrite("manodosky.png", manodosky) 




	#Convert PIL image format back to opencv image format
	opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
	if flag_texto >0:
		flag_texto = flag_texto - 1
		
		#contorno_quadra = np.array([[X_dir_alto,Y_dir_alto],[X_esq_alto,Y_esq_alto],[X_dir_baixo,Y_dir_baixo],[X_esq_baixo,Y_esq_baixo ]])

		verif_quique= cv2.pointPolygonTest( contorno_quadra[0], (x_quique,y_quique), False)
		if verif_quique>=0 :
			# Desenha o texto com a variavel em preto, no centro
			largura = 600
			altura = 200

			texto = 'Dentro'

			fonte = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
			escala = 2
			grossura = 3

			# Pega o tamanho (altura e largura) do texto em pixels
			tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)

			# Desenha o texto no centro
			cv2.putText(opencvImage, texto, (1700, 850), fonte, escala, (0, 0, 255), grossura)

		if verif_quique<0 :
			# Desenha o texto com a variavel em preto, no centro
			largura = 600
			altura = 400

			texto = 'Fora'

			fonte = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
			escala = 2
			grossura = 3

			# Pega o tamanho (altura e largura) do texto em pixels
			tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)

			# Desenha o texto no centro
			cv2.putText(opencvImage, texto, (1700, 850), fonte, escala, (0, 255, 0), grossura)
		
	#write image to output_video
	output_video.write(opencvImage)




	
# everything is done, release the video
video.release()
output_video.release()
print("finish")
print(conta_ai)
print(conta_total)

