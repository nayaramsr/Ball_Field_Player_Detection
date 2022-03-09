import argparse
import Models , LoadBatches
import tensorflow as tf
from keras import optimizers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#from tensorflow import ConfigProto
#from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Continuar com essa parte

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
parser.add_argument("--save_weights_path", type = str, default = "/content/gdrive/My Drive/PROJETO DOUTORADO/TrackNet/Code_Python3/TrackNet_Three_Frames_Input/weights/model" )    #)
parser.add_argument("--training_images_name", type = str, default = "training_model2.csv")                       #)
parser.add_argument("--n_classes", type=int, default = 256 )                                                     #)
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--epochs", type = int, default = 500)                                                       #1000 )
parser.add_argument("--batch_size", type = int, default = 1)                                                     #2 )
parser.add_argument("--load_weights", type = str , default = "-1")
parser.add_argument("--step_per_epochs", type = int, default = 200 )
############
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=2048)


args = parser.parse_args()
training_images_name = args.training_images_name
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
step_per_epochs = args.step_per_epochs
############
batch_size = args.batch_size
epochs = args.epochs

############
#optimizer_name = tf.keras.optimizers.Adadelta(lr=1.0)
optimizer_name = tf.keras.optimizers.SGD(0.01)


#load TrackNet model
modelTN = Models.TrackNet.TrackNet
m = modelTN( n_classes , input_height = input_height , input_width = input_width )
m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])

#check if need to retrain the model weights
if load_weights != "-1":
	m.load_weights("weights/model." + load_weights)

#show TrackNet details, save it as TrackNet.png
plot_model( m , show_shapes=True , to_file='TrackNet.png')

#get TrackNet output height and width
model_output_height = m.outputHeight
model_output_width = m.outputWidth

#creat input data and output data
Generator  = LoadBatches.InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)


#start to train the model, and save weights until finish 

#m.fit_generator( Generator, step_per_epochs, epochs )
#m.save_weights( save_weights_path + ".resnet.h5" )



#start to train the model, and save weights per 10 epochs  

for ep in range(1, epochs+1 ):
	print("Epoch :", str(ep) + "/" + str(epochs))
	m.fit(Generator, step_per_epochs)
	if ep % 10 == 0:
		m.save_weights( save_weights_path + ".resnet.h5" )
