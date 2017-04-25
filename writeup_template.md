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

Initially, I didn't apply any image pre-processing and decided to modify the network architecture to improve the network accuracy.

Once I was satisfied with the network channels of convolution, I applied normalization to the image pixel using the formula mentioned. ((pixel - 128)/128)

Following is the result -

![Normal][image10]
![Pre Processed][image11]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x15 	|
| RELU					| output 28x28x15								|			
| Max pooling	      	| 2x2 stride,  outputs 14x14x15 				|
| Convolution 3x3	    | 1x1 stride, same padding, output 12x12x15     |
| RELU                  | output 12x12x15 							    |
| Max pooling           | 2x2 stride, outputs 6x6x15                    |
| Flatten        	    | With 36x15 = 540 neurons                      |        					
| Fully connected 1		| 540 x 120        								|
| Fully connected 2		| 120 x 84			                            |											
| Fully connected 3		| 84 x 43				                        |									
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used AdamOptimizer to optimize the network and used batch size of 128 for training.
I trained the network for 50 EPOCHS.
The learning rate I used is 0.0005. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First I used the same LeNet architecture as described in the Udacity lecture.
One obvious thing was that this time we have more classes to categorize the data. 43 and not just 10.

So I increased the number of convolution channels in each layer significantly as compared to original LeNet.

I then tried to train the network for 10 EPOCHS and 0.001 learning rate.
But the network was jumping significantly in accuracy 
So I reduced the learning rate to 0.0005 and increased the EPOCHS to 50.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

Out of 6, I got 5 correctly recognized. One image failed possibly due to more zoom level or the watermark.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)	| Speed limit (50km/h) Incorrect  				|
| Speed limit (60km/h)	| Speed limit (60km/h)  						| 
| Speed limit (70km/h)	| Speed limit (70km/h)  						| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| No entry	      		| No entry					 			    	|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


