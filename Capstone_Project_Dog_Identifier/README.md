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

In this project, I used test accuracy to measure the performance of the models. Precision is used when the costs of false positives are high whereas recall is used when the costs of false negatives are high and F1 score is a metric to combine precision and recall. In our case, there is no false positives and false negatives. But accuracy tells us immediately whether a model is being trained correctly. So I wanted to know accuracy of our model by looking at test accuracy.

First of all I created CNN from scratch and used using transfer learning to train a CNN , with test accuracy of 6.9378%. <br/>

Secondly I used a CNN to Classify Dog Breeds from pre-trained VGG-16 model with test accuracy: 38.8756%. The second model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. <br/>

Third and last model is InceptionV3 bottleneck, with test accuracy of 80.2632%. <br/>

## Result Section :

When user upload an image, the model will first detect whether the image is human face or dog. If the model thinks it is a human it will tell "You are human", if the model thinks it is a dog, the model will try to predict the dog's breed. I tried with 3 human images and 3 dogs image. It worked with 100% accuracy.

## Conclusion

In this project, I have learned how to detect faces by using Neural Networks. I first started detecting human faces and then learned how to use Resnet-50 model to detect dog faces. As a next step I created a CNN from scratch to classify dog breeds. I also used transfer learning along with CNN to classify dog breeds. In this step, I used InceptionV3 model because Resnet50 and VGG19 models have over 20 million parameters that means have an enormous entropic capacity and are capable of just memorizing the training data images. During these steps I was able to see improvement of test accuracy up to 80%. In the last 2 steps I actually started using my trained model to test with providing images from my local computer randomly. I have given 6 images (3 human and 3 dogs) and model was able to predict all of them correct.
### Future Improvements:

There are several techniques for performance improvement with model optimization and one of them is using the model to predict on training data, retrain the model for the wrongly predicted images. This can be applied to our case to improve the test accuracy.

## Acknowledgments

I	am	grateful	to	all	of	those	with	whom	I	have	had	the	pleasure	to	work	during	this project.	Each of the	mentors	in this class	has	provided	me	extensive	guidance. 
