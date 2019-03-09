import numpy as np
from PIL import Image
import os
from numpy import linalg as LA
from keras.applications.vgg16 import VGG16 
from keras.layers import Dense, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import conv_utils,np_utils,plot_model
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import conv_utils,np_utils,plot_model
import argparse
import h5py
global input_shape,num_classes
from pathlib import Path

input_shape = (400,600,3)
num_classes = 16


class VGGnet:
	def __init__(self):
		self.input_shape= input_shape
		self.classes = num_classes
		self.weight='imagenet'
		self.pooling='max'
		self.model=VGG16(weights=self.weight,
			input_shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2]),
			pooling=self.pooling,
			include_top=False)

	def retrievalModel(self):

		for layer in self.model.layers:
			layer.trainable=False
		X= Dense(4096,activation='relu',name='fc7')(self.model.output)
		X= Dense(4096,activation='relu',name='fc8')(X)
		X= Dense(1024,activation='relu',name='fc9')(X)
		X= Dense(self.classes,activation='sigmoid',name='fc10')(X)
		model= Model(inputs=self.model.input,outputs=X,name='vgg16')
		return model 

	def get_data(self,folder_path,train_ratio):
		
		class_data = os.listdir(folder_path)
		train_x=[]
		train_y=[]
		test_x=[]
		test_y=[]

		class_number=0
		for classname in class_data:
			print(classname)
			images= os.listdir(os.path.join(folder_path,classname))

			class_size=len(images)
			test_count=0
			max_test_count= int((1-train_ratio)*class_size)
			count=0

			for img in images:

				temp=image.load_img(folder_path+'/'+classname+'/'+img,target_size=(400,600))
				temp = image.img_to_array(temp)
				temp = np.expand_dims(temp,axis=0)
				if count < max_test_count:
					test_x.append(temp)
					test_y.append(class_number)
				else:
			
					train_x.append(temp)
					train_y.append(class_number)
		
				count+=1
			class_number+=1

		train_x = np.vstack(train_x)
		print(train_x.shape)

		test_x = np.vstack(test_x)
		print(test_x.shape)

		train_y = np.array(train_y)
		test_y = np.array(test_y)

		return (train_x,train_y), (test_x,test_y)


	
	def train_model(self,folder_path,train_ratio,epochs,bs,lr,decay,mom,folder_path_to_save):

		(train_x,train_y),(test_x,test_y)=\
			VGGnet().get_data(folder_path,train_ratio)

		train_x=train_x/255
		test_x=test_x/255
		sgd = SGD(lr=lr, decay=decay, momentum=mom, nesterov=True)
		train_y=np_utils.to_categorical(train_y,self.classes)
		test_y=np_utils.to_categorical(test_y,self.classes)
		model= VGGnet().retrievalModel()
		model.compile(loss='binary_crossentropy',
			optimizer=sgd,metrics=['accuracy'])

		print("\n-------------------Training started-----------------------")

		model.fit(train_x,train_y,epochs=epochs,batch_size=bs)
		model.save_weights(os.path.join(folder_path_to_save,'vgg16.h5' ))

		print("\n-------------------Testing started-------------------------\n")
		loss,accuracy = model.evaluate(test_x,test_y)
		print("test loss:%f" % loss)
		print("test accuracy:%f" % accuracy)


def initialize_model(path_to_pre_weights):

	model = VGGnet().retrievalModel()
	model.load_weights(path_to_pre_weights)
	return model

def  main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset",required = True, help="Path to dataset")
	ap.add_argument("--output",required=True, help="Folder Path to save")
	ap.add_argument("--epochs",required =True, type=int,help="Number of epochs")
	ap.add_argument("--bs",required=True,type=int, help="Batch Size")
	ap.add_argument("--lr",required=True,type=float, help="Learning rate")
	ap.add_argument("--decay",required=True,type=float, help="Decay")
	ap.add_argument("--mom",required=True,type=float, help="Momentum")
	ap.add_argument("--ratio",required=True,type=float, help="Train-Test ratio")
	args = vars(ap.parse_args())

	VGGnet().train_model(args["dataset"],args["ratio"],args["epochs"],
						args["bs"],args["lr"],args["decay"],args["mom"],
						args["output"])




if __name__=='__main__':

	main()

