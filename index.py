
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
import h5py
from CNN import VGGnet
import argparse



def extract_features(image_path):
	img = image.load_img(image_path,target_size=(400,600))
	img = image.img_to_array(img)
	img = np.expand_dims(img,axis=0)
	img = preprocess_input(img)



	intermediate_layer_model = Model(input=model.input,
									outputs=model.get_layer('fc7').output)
	feat = intermediate_layer_model.predict(img)	
	return feat


def create_index_file(folder_path,path_to_pre_weights,output):
	class_data = os.listdir(folder_path)

	features = []
	names = []

	model = VGGnet().retrievalModel()
	model.load_weights(path_to_pre_weights)
	intermediate_layer_model = Model(input=model.input,
								outputs=model.get_layer('fc7').output)
	
	for classname in class_data:
		print(classname)
		filenames = os.listdir(os.path.join(folder_path,classname))
		count=0

		for file in filenames:			
			img = image.load_img(os.path.join(folder_path,
											classname,file),target_size=(400,600))
			img = image.img_to_array(img)
			img = np.expand_dims(img,axis=0)
			img = preprocess_input(img)

			feat = intermediate_layer_model.predict(img)
			norm_feat = feat[0]/LA.norm(feat[0])
			features.append(norm_feat)
			names.append(classname+'_'+file)
			count+=1

	output = os.path.join(output,"index.hdf5")
	index_file = h5py.File(output,'w')
	index_file.create_dataset('dataset_feat',data=np.array(features))
	index_file.create_dataset('dataset_name',data=np.array(names))
	index_file.close()

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required = True, help = "Path to dataset to be indexed")
	ap.add_argument("--output", required = True, help = "Path to output")
	ap.add_argument("--pre", required = True, help = "Path to pretrained weights")

	args = vars(ap.parse_args())
	create_index_file(args["dataset"],args["pre"],args["output"])


if __name__ == '__main__':

	main()

