# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode, here I've changed the set speed from 9 MPH to 20 MPH, and converted the input images to grayscale.
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The images are cropped in the model using a Keras cropping2D layer, to remove the irrelevant portions at the top and bottom of the images (trees, car hood, etc). Then the data is normalized using a Keras lambda layer (code line 80).

My model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 81-89).

The model includes RELU layers to introduce nonlinearity (code lines 82, 84, 86, 88, 90, 94, 97, 100).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 93, 96, 99). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and left / right camera data with a steering angle correction factor to emulate recovering from the sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different well known models, and optimize them to work well for the problem at hand.

My first step was to use a convolutional neural network model similar to the Lenet model, I thought this model might be appropriate because it is simple enough, yet it is proven effective for recognizing simple shapes/edges etc.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has a dropout layer for each fully connected layer, placed before the RELU activation, with a 0.3 keep probility.

I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially near the dirt bit where there is no road edge marking, to improve the driving behavior in these cases, I did both clockwise and counterclockwise training laps which helped greatly to solve this issue.

At the end of the process, the vehicle was able to drive autonomously around track one without leaving the road.

However for track two, the Lenet model was not sufficient, since it has much heavier turns, shadows, ups and downs, so I needed a bigger more powerful network that can recognize more features in the training data.

So at the end I changed the model to make it similar to the model NVIDIA uses for End to End driving, which worked well for both track one and two.

#### 2. Final Model Architecture

The final model architecture (model.py lines 78-102) consisted of a convolutional neural network with the following layers and layer sizes:

* Cropping Layer.
* Lambda Layer for normalizing data.
* 5x5 Convolutional layer with (2, 2) stride, depth of 24 and RELU activation.
* 5x5 Convolutional layer with (2, 2) stride, depth of 36 and RELU activation.
* 5x5 Convolutional layer with (2, 2) stride, depth of 48 and RELU activation.
* 3x3 Convolutional layer with depth of 64 and RELU activation.
* 3x3 Convolutional layer with depth of 64 and RELU activation.
* Flatten layer.
* 100 node Fully connected layer with 0.3 prob dropout and RELU activation.
* 50 node Fully connected layer with 0.3 prob dropout and RELU activation.
* 10 node Fully connected layer with 0.3 prob dropout and RELU activation.
* Single node output layer generating the steering angle.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data set, I flipped images and angles which helped balance the heavy pulling to the left side of the road.

![alt text][image6]
![alt text][image7]

I then recorded the vehicle recovering from the left side and right sides of the road back to center, but I found that this approach is cumbersome, not realistic and doesn't make much difference anyway since I am already using the data from the left and right cameras to emulate this driving behavior, so I discarded this data. Instead I chose to record a 3rd lap to give the model more data to train on.

I then found that recording laps going the opposite way (clockwise) is very helpful to make the model generalize better and solves the issue of falling off the track in some areas, so I recorded another 3 laps going the opposite direction, making a total of 3 forward laps and 3 reverse laps.

Then I repeated this process on track two in order to get more data points, and enhance the driving behavior in track two (track one is already pretty good at this point).

I found that converting the images to grayscale was beneficial in track two, since the colors of the road edge markings and surroundings are different between the 2 tracks so ignoring the colors all together leads to better generalization.

But still, at this point the car was able to barely go through 5% of track two, so I thought maybe the model is challenged by the heavy shadows (can't see well in shadows), so to solve this problem I decided to augment the data by adding a low brightness copy of all the images, to train the model to see in low light conditions (multiplying the grayscale images by a factor of 0.2 (model.py line 60))

Indeed, this solved the problem, and the car was able to smoothly go through 95% of track two, except for a difficult bit which I had to go back and collect more data specific to this bit, to be able to 100% finish track two.

After the collection process, I had X number of data points, 20% of this data was split into a validation set, and a generator was used to output batches of shuffled data to the model at a time (to save GPU memory).

The data are further preprocessed in the Keras model itself, by cropping the irrelevant upper and lower portions of the images and normalizing the date.

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by the graph showing the training and validation loss, which shows the that the model kind of plateaus after 30 epochs, 50 epochs will lead to better results, but is not worth the extra training time. I used an adam optimizer so that manually training the learning rate wasn't necessary.
