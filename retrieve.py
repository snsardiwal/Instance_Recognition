import numpy as np
import h5py
import os
from CNN import VGGnet
from index import extract_features
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from numpy import linalg as LA
from keras.applications.vgg16 import VGG16 
import argparse
from CNN import initialize_model



def extract_instance(instance_path):
	f = open(instance_path,'r')
	instances={}
	for line in f:
		fields=line.split(" ")
		instances[fields[0]] = fields[1].split(',')

	return instances


def online_query(query_path,index_path,rank_path_to_save,model):
	
	index_file = h5py.File(index_path,'r')
	feats = index_file['dataset_feat'][:]
	names = index_file['dataset_name'][:]
	index_file.close()

	intermediate_layer_model = Model(input=model.input,
								outputs=model.get_layer('fc7').output)

	query_img = image.load_img(query_path,target_size=(400,600))
	query_img= image.img_to_array(query_img)
	query_img = np.expand_dims(query_img,axis=0)
	query_img = preprocess_input(query_img)

	query_feat = intermediate_layer_model.predict(query_img)
	norm_feat = query_feat[0]/LA.norm(query_feat[0])
	norm_feat = np.array(norm_feat).reshape(4096,1)

	feats = np.squeeze(feats)

	scores = np.dot(norm_feat.T,feats.T)
	rankID = np.argsort(scores)[::-1]

	fields=query_path.split('/')
	query = fields[-1].split('.')

	f = open(os.path.join(rank_path_to_save,query[0] + '.txt'),'a')
	imlist = [names[index] for i,index in enumerate(rankID)]
	imlist = np.squeeze(np.array(imlist))
	
	for i in range(3456):
		f.write('%s\n' % imlist[3455-i])
	f.close()

def loop(query_path,index_path,path_to_pre_weights,rank_path_to_save):
	model = initialize_model(path_to_pre_weights)
	for dirName, subdirList, fileList in os.walk(query_path):
		print(dirName,subdirList,fileList)
		for file in fileList:
			print(file)
			online_query(os.path.join(query_path,file),index_path,rank_path_to_save,model)
		break


"""
def create_ranklist(query_path,instance_path,index_path,
					path_to_pre_weights, rank_path_to_save):
	instances = extract_instance(instance_path)
	images = list(instances.keys())
	p={}
	model = initialize_model(path_to_pre_weights)
	for img in images:
		p[img]=0
		classnames=instances[img]
		online_query(os.path.join(query_path,img),index_path, rank_path_to_save,model)

"""


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--query", required=True, help = "Path to query images")
	# ap.add_argument("--instances", required=True, help = "Path to instance file")
	ap.add_argument("--index", required=True, help = "Path to index file")
	ap.add_argument("--pre", required=True, help = "Path to pretrained weights")
	ap.add_argument("--rank", required=True, help = "Path to output dir containing rank files")
	
	args= vars(ap.parse_args())

	#model=initialize_model(args["pre"])
	print("entered")
	loop(args["query"],args["index"],
					args['pre'],args["rank"])


if __name__ == '__main__':

	main()
	
