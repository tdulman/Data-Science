# Dog Identificator App
## Project Overview

Our code will try to identify human faces and dog's breed at the end of this project.

## Project Overview

I completed couple of project about Data Science and Machine Learning. This is my Capstone Project to complete my Data Science Nano Degree Program.

## Steps of Project

I followed these steps to complete the project.

Step 0: Import Datasets
Step 1: Detect Humans
Step 2: Detect Dogs
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
Step 6: Write your Algorithm
Step 7: Test Your Algorithm


## File Structure

bottleneck_features: Stores bottleneck features
dog_app.ipynb : Codes and possible solutions can be found.
images : Stores human and dog images.
saved_models - Stores models I worked on

## Required Libraries

NumPy, Pandas, matplotlib, scikit-learn, keras, OpenCV, Scipy, Tqdm, Pillow, Tensorflow, Skimage, IPython Kernel

## Improvements Section

First I created CNN from scratch and i use using transfer learning to train a CNN , with test accuracy of 6.1005%. 

Second I used a CNN to Classify Dog Breeds from pre-trained VGG-16 model with test accuracy: 38.1579 %.

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

Third I then used Transfer learning to create a CNN that can identify dog breed from images with 80.1435 % accuracy on the test set.

My final CNN architecture is built with the Resnet50 bottleneck. Further, GlobalAveragePooling2D used to flatten the features into vector. These vectors were fed into the fully-connected layer towards the end of the ResNet50 model. The fully-connected layer contains one node for each dog category and is assisted with a softmax function.

## Result Section :



