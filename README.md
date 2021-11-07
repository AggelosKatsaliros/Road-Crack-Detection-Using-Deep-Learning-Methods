# Road-Crack-Detection-Using-Deep-Learning-Methods
  This is my Diploma Thesis ¨Road Crack Detection Using Deep Learning Methods¨ under the supervision of             Dr.George Sfikas.

## Technologies
* Python
* Pytorch
* LaTex

### Libraries
* numpy
* sklearn
* PIL
* splitofolders
* torch
* time
* torchvision
* tqdm
* [quaternion library](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks)


## Intro
  Road crack detection has vital importance in driving safety. However, it is
  very challenging because of the complexity of the background, cracks are
  easily confused with foreign objects, shadows, background textures and are
  also inhomogeneous. Many methods have been proposed for this task, but
  Convolutional Neural Networks(CNN) are promising for crack classification
  and segmentation with high accuracy and precision. Another method that
  is being studied is the use of quaternions. Quaternions have the advantage
  of providing more structural information of the color and as a result offer
  better learning results and avoid overfitting. In this study we focused on
  implementing various cnn networks and compare their results with Quaternion Convolutional Neural Networks         (QCNN) which are an extension of the
  cnn in the quaternion domain. Specifically we replaced the convolutional
  and linear layers of the cnn’s with quaternion convolutional layers and linear
  layers respectively.
  
  
## Structure
Our Project has two components, the classification and the segmentation part. In the classification part we build neural networks such as Alexnet ,Vgg16 and a custom one and in the segmentation part we took an already implemented Deep Hierarchical Feature Learning Architecture for Crack Segmentation [DeepCrack](https://github.com/yhlleo/DeepCrack).
<img

## Dataset
  For the classification part we took images from [here](https://data.mendeley.com/datasets/xnzhj3x8v4/2) and [here](https://drive.google.com/drive/folders/1oJ-yoOaUf2TPbUB1LznrHOas_7imd68o)

