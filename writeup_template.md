#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is using NVIDIA architecture and I found it perform pretty well. As a data wrangling using Cropping 2d (converting images from (160, 320,3) into (76, 320, 3)) and lambda layer for normalization (converting every pixel value in the range of -0.5 to 0.5)

Code snippet to show model structure as below:

```
    model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
```

The model includes RELU layers to introduce nonlinearity, and the data is normalized using a Keras lambda layer. Cropping is also introduced to shift more focus on the road itself(top 70 and bottom 25 pixels are cropped).

At first ELU is used for the nonlinearity layer. However, after trying RELU turns out to have a better performance. 


####2. Attempts to reduce overfitting in the model

The model contains 1 dropout layer in order to reduce overfitting which is before the flattening layer to make sure the validation loss is decreasing.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

```
X_train, X_validation, y_train, y_validation = train_test_split(X_sample, y_sample, test_size=0.2)

```

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

In addition to the dataset provided by Udacity, I created some additional datasets for specific senarios to enhance the model:

* Dataset for all the curves and turns in the track.

* Dataset for all the curves and turns in the track by driving the car in the opposite direction.

* Dataset for the straight roads in the track.

Moreover, because the dataset I created are much smaller than the original dataset, I duplicate the turnning dataset to enhance driving in that area. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step is to use the NVIDIA model structure and the original dataset to train the model. The original model performs well on straight lane but cannot turn well, especially when there are trees in the picture. I then added the cropping layer to remove distracting parts of the picture, and created several additional training datasets specifically for turns. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Also, I tuned some of the parameters: cropping size and correction for left/right cameras. For cropping size, I tried 60, 70 or 80 for upper side, and turns out 70 is the best. For correction, I also tried 0.1, 0.2, 0.3 and 0.4, and turns out 0.3 is the best. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes. Three 55 layers with subsampling. and two 3 3 layers. I added a dropout layer of 0.5 to avoid overfitting. Here is a visualization of the architecture(Nvidia model without dropout layer)

![alt text](https://github.com/jingxia/CarND-Behavioral-Cloning-P3/blob/master/examples/model_structure.png)


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the original dataset provided by udacity. 

Because that the model doesn't perform very well on turns, I then recorded 4 additional datasets for specific turns of track 1. My datasets are much smaller than the original dataset, so I manually duplicate them so that these inputs have enough weight in the training data to improve the model. 

For each dataset, I take all center, left and right images. A correction of 0.3 is applied to left or right images. Also each image is flipped so that it can handle both right or left turns. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
