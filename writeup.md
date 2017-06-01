# **Behavioral Cloning** 


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
[img_hist]: ./charts/hist.png "Histogram"
[img_afterdown]: ./charts/afterdown.png "After downsample"
[img_video]: ./charts/steer.png "Youtube link"
[img_jitter]: ./charts/jitter.png "jitter"
[img_left]: ./charts/left_cam.jpg "left"
[img_center]: ./charts/center_cam.jpg "center"
[img_right]: ./charts/right_cam.jpg "right"
[img_noflip]: ./charts/steer_noflip.jpg "noflip"
[img_flip]: ./charts/steer_flip.jpg "flip"
[img_translate]: ./charts/translate.png "translate"
[img_notranslate]: ./charts/no_translate.jpg "notranslate"
[img_resize]: ./charts/resized.png "resize"
[img_track2]: ./charts/track2.png "track 2"
  

---
### Files Submitted & Code Quality

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

To train the model, execute ```python model.py --training_file=<training csv> --output=<model_file>```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

I used LeNet as a basis for expanding/altering the model depending on the validation error I get, and the observations of the error the model made while steering on the track.

Later due to a programming bug, for debugging purposes, I imitated the model used [here](https://medium.com/m/global-identity?redirectUrl=https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) for some time to ensure I get enough representation power, as I was not getting progress despite trying out various suggestions from the Udacity forum.

(It turned out that I was using cv2.imread() which arrange the image array in BGR format during training, and drive.py reads with PIL in RGB during simulation drive. Garbage IN, garbage OUT).

Once the error was resolved, I then progressively scaled down the network architecture by reducing the number of filters/nodes and layers needed. This helped in preventing overfitting as the number of parameters/weights is significantly lowered.

The final archictecture looked like this:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Crop         		| Crop off top 30 and bottom 25 pixels to remove the steering wheel and horizon   							| 
| Preprocess / lambda         		| Zero-mean features and rescaled to 0-1. Resized image to 64 by 64 pixels   							|    							| 
| Convolution kernel 5x5     	| 1x1 stride, VALID padding, 32 filters 	|
| Activation					| Relu												|
| Max pooling 2x2	      	| 2x2 stride 				|
| Convolution kernel 5x5	    | 1x1 stride, VALID padding, 64 filters      									|
| Activation | Relu |
| Max pooling 2x2 | 2x2 strides, VALID padding
| Fully connected		| 60 nodes        									|
| Activation | Relu
| Dropout | p=0.5
| Fully connected				| 32 nodes        									|
| Activation | Relu
| Dropout | p=0.5
| output						| 1 node (output == steering angle)  												|
|						|												|

#### 2. Attempts to reduce overfitting in the model

The model uses 3 strategies to combat overfitting:

* Dropout in the fully connected layers
* Image augmentation. See section on Training Set creation.
* Reducing the number of parameters by using smaller networks.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


#### 4. Creation of the Training Set & Training Process

The dataset used for training and validation purposes is obtained from the sample data provided by Udacity solely. This consist of 24108 captured images from the left, right and center cameras, together with the recorded steering angle.

No further attempts at data collection through recording was used. This meant that recovery steering back to the center of the lane from off positions will be weak, and should be required in end-to-end real life driving (possibly from avoidance of obstacles etc). But for the purpose of this simulation drive, it will force the learnt model to be really stringent on staying on the lane and correcting as early as possible.

Because the number of data points are relatively small (24108/3=8036), there are a number of ways I did to increase the training sample sizes

* flip the images. This turns out to be an important step. Because the training track has more left turns than right, there is a real bias for the model to steer left.

| Angle         		|     Image	        					| 
|:---------------------:|:---------------------------------------------:| 
|  -0.0617599         		|   		![alt text][img_flip]
|  0.0617599				|			![alt text][img_noflip]

* Left/right camera images. Because the cameras are "installed" on the left and right sides of the vehicle, they can be augmented (by adding an offset to the steering angle. I use 0.2 in the model) to be viewed as the center camera when the vehicle is off-center). These provides excellent data points for slight recovery steering. 

| Camera         		|     Image	        					| 
|:---------------------:|:---------------------------------------------:| 
| Left         		|   		![alt text][img_left]
| Center				|			![alt text][img_center]
| Right				|			![alt text][img_right]
										

* left and right translation. Inspired by another [student](https://medium.com/m/global-identity?redirectUrl=https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9), images are left or right shifted by a maximum of <-25,25> pixels. This provide slight variations from the original image, which helps in providing overfitting. However, instead of adopting the per pixel correction the author used, I have instead retained the original steering angle, as I do not believe the position of the vehicle relative to the lane has changed after the shift.

| Offset         		|     Image	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0         		|   	![alt text][img_notranslate]
| -25				|			![alt text][img_translate]

* Color jittering.

![alt text][img_jitter]

As mentioned earlier in the Network Architecture section, after augmentating the images, the following preprocessing is done before feeding into the network

1. Cropping off top 40 and bottom 25 pixels
2. Scaling values between 0-1 and 0 mean
3. Resize to 64 by 64 pixels.

Here is n sample without any augmentation:

![alt text][img_resize]

I finally randomly shuffled the data set and put 10% of the data into a validation set. I used this training data for training the model. Only a small amount of validation set was used, to maximize the number of training examples. While the validation error is a good indication of overfitting/underfitting, poor model and program bugs and whether the model will do well (empircal observations seem to suggest < 0.03 threshold), the eventual metric/goal is that the model completes the track. A noteworthy observation is that a model with 0.01 validation rms does not necessary do better than another with 0.02 error. 

I trained the model for a maximum of 5 epochs. Beyond 5 iterations, while minimal improvements can be made to the validation error, it does not seem to help towards completing the track.
 
![alt text][img_hist]
As seen from the histogram above, there is a strong bias towards small angle steering. Even after adding the left and right camera images, there is still biases towards 0, `0 + correction`, `0 - correction `degrees.

To mitigate this, I downsampled the small angle records

~~~~
rand = random.random()
if rand >= 0.2 and abs(steer) <= 0.2:
	continue   
~~~~

The effects can be visualized with the weighted histogram, after adding the left and right camera data points. While still not well balanced, we can see that the data is less skewed towards 0 degree already. 
![alt text][img_afterdown]


### Results and Discusssions

The trained model, despite having only 2 convolutional and 2 fully connected layers, worked well in the track 1. You can view the drive video by clicking on the link below. The viewing perspective in the video is taken from the snapshot of the center camera, and thus the image used for prediction as well.

[![alt text][img_video]](https://www.youtube.com/watch?v=g6-oCpgJjSQ)

Track 2, however didn't perform too well. The vehicle did not make it pass the turn up the first slope. It seem that with the model trained entirely on track 1, the model has mistakenly recognize the center lane marking as the road boundary, which explains why it did not try to correct itself when it was entirely on the left lane. Obviously this isn't wrong, which itself is actually the correct behavior. However, the model hasn't learnt to steer hard enough on the tighter lane before it went off the road.

Therefore further possible work that might enhance better track 2 performance would be to record off position scenarios near bends and build it on top of the existing model.   

[![alt text][img_track2]](https://youtu.be/Mx4vJbdUN4U)



