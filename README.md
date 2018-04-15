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
[image2]: ./writeup/center.jpg "Center Driving"
[image3]: ./writeup/normal.jpg "Normal Image"
[image4]: ./writeup/flip.jpg "Flipped Image"
[image5]: ./writeup/gray.jpg "Grayscale"
[image6]: ./writeup/graylb.jpg "Low Brightness"
[image7]: ./writeup/loss.png "Loss Graph"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the vehicle in autonomous mode, here I've changed the set speed from 9 MPH to 20 MPH, and converted the input images to grayscale.
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* track1_run.mp4 video file showing a successful autonomous lap around track one.
* track2_run.mp4 video file showing a successful autonomous lap around track two.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the vehicle can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 85-93).

The model includes RELU layers to introduce nonlinearity (code lines 86, 88, 90, 92, 94, 98, 101, 104).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 97, 100, 103). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21). The model was tested by running it through the simulator and ensuring that the vehicle could stay on both tracks.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 106).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I solely used center lane driving with the aid of left / right cameras data to emulate recovering from the sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different well known models, and optimize them to work well for the problem at hand.

My first step was to use a convolutional neural network model similar to the Lenet model, I thought this model might be appropriate because it is simple enough, yet it is proven to be effective for recognizing simple shapes/edges etc.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has a dropout layer for each fully connected layer, placed before the RELU activation, with a 0.3 keep prabability.

I ran the simulator to see how well the vehicle was driving around track one. There were a few spots where the vehicle fell off the track, especially near the dirt bit where there is no road edge marking, this problem was solved by collecting more data which will be shown below.

At the end of the process, the vehicle was able to drive autonomously around track one without leaving the road.

However for track two, the Lenet model was not sufficient, since it has much heavier turns, shadows, ups and downs, so I needed a bigger more powerful network that can recognize more features in the training data.

So at the end I changed the model to make it similar to the model NVIDIA uses for it's End to End driving, which worked well for both track one and two.

#### 2. Final Model Architecture

The final model architecture (model.py lines 82-106) consisted of a convolutional neural network with the following layers and layer sizes:

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

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data set, I flipped images and angles which helped counteract the heavy pulling to the left side of the road.

![alt text][image3]
![alt text][image4]

I then recorded the vehicle recovering from the left and right sides of the road back to center, but I found this approach cumbersome, not realistic and doesn't make much difference since I am already using the data from the left and right cameras to emulate this driving behavior, so I discarded this data. Instead I chose to record a 3rd lap to give the model more data to train on.

I then found that recording laps going the opposite way (clockwise) is very helpful to make the model generalize better and solves the issue of falling off the track in some areas, so I recorded another 3 laps in the opposite direction, making a total of 3 forward laps and 3 reverse laps.

Then I repeated this process on track two in order to get more data points, and enhance the driving behavior in track two (track one was already pretty good at this point).

I found that converting the images to grayscale was beneficial in track two, since the colors of the road edge markings and surroundings are different between the 2 tracks, so ignoring the colors altogether leads to better generalization.

![alt text][image5]

But still, at this point the vehicle was barely able to go through 5% of track two, so I thought maybe the model is challenged by the heavy shadows (can't see well in shadows), so to solve this problem I decided to augment the data by adding a low brightness copy of all the training images to the training dataset, to train the model to see in low light conditions (multiplying the grayscale images by a factor of 0.2 (model.py line 61))

![alt text][image6]

Indeed, this solved the problem, and the vehicle was able to smoothly go through 95% of track two, except for a difficult bit at the end, which made me go back and record additional 2 forward laps and 2 reverse laps (only for track two) which made the vehicle successfully finish 100% of track two.

After the collection process, I had 75,534 number of data points, 20% of this data was split into a validation set, and a generator was used to output batches of shuffled training and validation data to the model at a time, in order to save GPU memory.

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was found to be 30 as evidenced by the below graph of the training and validation loss, which shows that learning almost plateaus after 30 epochs, indeed 50 epochs will lead to better results, but it is not worth the extra training time. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image7]

The data are further preprocessed in the model itself, by cropping the irrelevant upper and lower portions of the images and normalizing the data.
