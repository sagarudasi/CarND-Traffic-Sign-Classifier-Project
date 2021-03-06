#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_set_vis.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/60.jpg "60 miles per hour"
[image5]: ./examples/60_2.jpg "60 miles per hour"
[image6]: ./examples/70.jpg "70 miles per hour"
[image7]: ./examples/stop.jpg "Stop"
[image8]: ./examples/yield.jpg "Yield"
[image9]: ./examples/noentry.jpg "No entry"
[image10]: ./examples/normal_image.png "Normal image"
[image11]: ./examples/pre_processed.png "Pre-processed"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sagarudasi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate the shapes of different set of data and following is the summary:

* The size of training set : 34799
* The size of the validation set : 4410
* The size of test set : 12630
* The shape of a traffic sign : 32x32 with 3 channels (RGB)
* The number of unique classes/labels in the data set : 43

####2. Include an exploratory visualization of the dataset.

Below is the visualization of the training data set. Basically, we have 43 different classes. So I counted the number of training examples of each class and then plotted it on y axis.

![Total class and number of training examples per class][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I used two pre-processing techniques here - Converting images to grayscale and normalization 

***Grayscaling***
The original images are colored images. It consists of three channels red, green and blue. For identifying traffic signs, identifying and training the network based on shapes will work just as fine as with shapes and RGB colors.
By converting images to grayscale, we are basically reducing the problem to 1 dimension and the network will get trained to learn shape features that comprises of curves and lines instead of colors.

The formula I used is following - grayvalue = 0.299 * RED + 0.587 * GREEN + 0.114 * BLUE

![Original image][image10]

After converting to grayscale
![Grayscaling image][image11]

***Normalization***
Normalization involves converting data such that the values are centered towards average/mean. It helps in converging the values as the contours are circular. Otherwise, if values are not properly distrubuted, the contours may huge oval shapes.
The original data values are between 0-255 so I used the following formula to normalize the values - pixelvalue = (pixelvalue - 128) / 128.

The final values are between -1 and 1 as following - 
-0.33865625 -0.43377344 -0.53725781 ..., -0.34489844 -0.34472656 -0.246

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Pre-processing        | 32x32x1 grayscale image   					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU					| output 28x28x10								|			
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 3x3	    | 1x1 stride, same padding, output 12x12x18     |
| RELU                  | output 12x12x18 							    |
| Max pooling           | 2x2 stride, outputs 6x6x18                    |
| Convolution 3x3	    | 1x1 stride, same padding, output 6x6x26       |
| RELU                  | output 6x6x26 							    |
| Flatten        	    | With 16x28 = 416 neurons                      |        					
| Fully connected 1		| 416 x 120        								|
| Dropout				| 50%                                           |
| Fully connected 2		| 120 x 43				                        |									



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used AdamOptimizer to optimize the network. Batch size of 128 was used for training.
I trained the network for 100 EPOCHS but then realized that it was not improving much beyond 94% accuracy on validation set. 
In order to avoid overfitting, finally I reduced the EPOCHS to 30 and the accuracy was 93.4 on validation set.
Learning rate of 0.001 was working fine. I tried reducing it to half but not much of improvement and error reduced almost properly with 0.001.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First I used the same LeNet architecture as described in the Udacity lecture.
One obvious thing was that this time we have more classes (43 and not 10) to categorize the data and thus more features to be extracted.

I tried to train the network for 10 EPOCHS and 0.001 learning rate and compared the training accuracy with the validation accuracy.
I observed the network was underfitting.

So I decided to add one more convolution layer and increased the channels further.

This time the network started overfitting. So I added 50% dropout and also removed one fully connected layer.

Then I ran the program for 100 EPOCHS and got max accuracy of 95.2 at about 94th EPOCH which was acceptable.
But then as it was not improving much above 94%, in order to avoid overfitting, I reduced the EPOCHS to 30.

My final model results were:
* Train accuracy: 0.988 
* Validation accuracy: 0.934 
* Test accuracy: 0.913


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Following are the images which I found on net and converted to 32x32 JPG using GIMP tool.

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]

Out of 6, I got 5 correctly recognized. One image failed possibly because its slightly oval in shape than round and also trees in background. However, it is detecting that image in same "sub-class", i.e. its a speed limit image.  
Other images are detected accurately.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (60km/h)	| Speed limit (60km/h)   				        |
| Speed limit (60km/h)	| Speed limit (120km/h)  	Incorrect			|
| Speed limit (70km/h)	| Speed limit (70km/h)  					    |
| Yield     			| Yield 										|
| Stop					| Stop											|
| No entry	      		| No entry					 			    	|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is working with acceptable accuracy on the images which I downloaded.
Consider the first image, it was recognized as Speed limit 60km/h with 99.9% accuracy which is correct.

 The top five soft max probabilities for first image were -

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99147058e-01        | Speed limit 60km/h   							|
| 8.18604894e-04     	| Speed limit 50km/h							|
| 3.41815758e-05		| Wild animals crossing							|
| 1.17701603e-07	    | Speed limit (30km/h) 		                    |
| 1.75457556e-14		| Speed limit (80km/h)      					|

Here we can see that most of the top predicted images belong to the same sub-class of the sign (Speed limit). 
Hence we can say that the network is able to generalize the features of traffic sign.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Not implemented
