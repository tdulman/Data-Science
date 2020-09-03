# Dog Identificator App
## Project Overview

Our code will try to identify human faces and dog's breed at the end of this project.

## Project Overview

I completed couple of project about Data Science and Machine Learning. This is my Capstone Project to complete my Data Science Nano Degree Program.

## Steps of Project

I followed these steps to complete the project.

Step 0: Import Datasets <br/>
Step 1: Detect Humans <br/>
Step 2: Detect Dogs <br/>
Step 3: Create a CNN to Classify Dog Breeds (from Scratch) <br/>
Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning) <br/>
Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning) <br/>
Step 6: Write your Algorithm <br/>
Step 7: Test Your Algorithm <br/>


## File Structure

bottleneck_features: Stores bottleneck features <br/>
dog_app.ipynb : Codes and possible solutions can be found. <br/>
images : Stores human and dog images. <br/>
saved_models - Stores models I worked on <br/>

## Required Libraries

NumPy, Pandas, matplotlib, scikit-learn, keras, OpenCV, Scipy, Tqdm, Pillow, Tensorflow, Skimage, IPython Kernel

## Improvements Section

First of all I created CNN from scratch and used using transfer learning to train a CNN , with test accuracy of 6.9378%. <br/>

Secondly I used a CNN to Classify Dog Breeds from pre-trained VGG-16 model with test accuracy: 38.8756%. The second model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. <br/>

Third and last model is InceptionV3 bottleneck, with test accuracy of 80.2632%%. <br/>


## Result Section :

When user upload an image, the model will first detect whether the image is human face or dog. If the model thinks it is a human it will tell "You are human", if the model thinks it is a dog, the model will try to predict the dog's breed. I tried with 3 human images and 3 dogs image. It worked with 100% accuracy.

