# Fine grained classification using BCNN
Multiple instance recognition using VGG16(fc7) as a feature extractor.

### Introduction:
The features are summarized below:
+ Use VGG16 to extract features (fc7).Firstly, the network is trained on the training dataset then for every image in the dataset, feature vectors(output of fc7) are extracted and saved in a index file. During retrieval phase, feature vector of query image is obtained and similarity with feature vectors(stored in index file) of all the images is obtained using cosine similarity. A text file is created which gives the ranking order of images in dataset in decreasing order.

### Usage:
```
+ Train the model
```
CUDA_VISIBLE_DEVICES=1 python2 train.py --dataset Data/train --output pretrained --epochs 10 --bs 8 --lr 0.05 --decay 1e-5 --mom 0.9 --ratio 0.7


```
+ Create Index file
```
 CUDA_VISIBLE_DEVICES=3 python2 index.py --dataset paht/to/train/data --output path/to/dir/to/save --pre path/to/saved/model
```

+ Retrieve Images
```
 CUDA_VISIBLE_DEVICES=3 python2 retrieve.py  --query path/to/query/data  --index path/to/index/file --pre path/to/pretrained/weights --rank path/to/dir/to/store/output

 ```
```

```

