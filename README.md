# Computer Vision System Applied in Tennis Games Analysis

In order to make the ball tracking, we used a CNN based algorithm called TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks proposed on: 

1. Yu-Chuan Huang, "TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks," 
    Master Thesis, advised by Tsì-Uí İk and Guan-Hua Huang, National Chiao Tung University, Taiwan, April 2018.
2. Yu-Chuan Huang, I-No Liao, Ching-Hsuan Chen, Tsì-Uí İk, and Wen-Chih Peng, 
    "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications,"
    in the IEEE International Workshop of Content-Aware Video Analysis (CAVA 2019)
    in conjunction with the 16th IEEE International Conference on Advanced Video and Signal-based Surveillance (AVSS 2019),
    18-21 September 2019, Taipei, Taiwan.
    
    
For the cnn network to work, follow the steps below:


#### First, you have to install cuda, cudnn and tensorflow, tutorial:
https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e


#### Second, install some python library with pip:
* sudo pip install numpy
* sudo pip install matplotlib    #Atual: python3 -m pip install matplotlib
* sudo pip install pillow
* sudo pip install keras         #Atual: sudo pip3 install keras
* sudo pip install opencv-python #Atual: python -m pip install opencv-python
* sudo pip install pydot
* sudo pip install h5py
* sudo apt-get install graphviz
    

## Detection of ball, field and player position with an input video:
1. Open command line
2. Change directory to TrackNet folder (TrackNet_Three_Frames_Input)
3. using following command as example, you may need to change the command:
	
		python  Ball_Field_Player_Detection.py  --save_weights_path=weights/model.3 --input_video_path="/home/nayara/Documentos/TrackNet/Code_Python3/TrackNet_Three_Frames_Input/Clip1.mp4" --output_video_path="/home/nayara/Documentos/TrackNet/Code_Python3/TrackNet_Three_Frames_Input/Clip1_TrackNet.mp4" --n_classes=256 

	* Detailed explanation
			--save_weights_path: which model weight need to be loaded
			--input_video_path: Input video path
			--output_video_path: Output video path, if not, the video will be save in the same path of input video
			--n_classes: In this work depth be set as 256 
