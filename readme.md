# My Notes Regarding Machine Vision 

## Convolution Filter Edge Detection: 

### 1. Fouriour Transform:
* Fouriour Transformation is used in CV to transform Image(Spatial) domain to freq domain
* it can help to visualize changes(Freq) in the images 

[Example](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/1.%20Fourier%20Transform.ipynb)

### 2. Canny Edge Detector: 
Widely used, accurate edge detection algorithm

internally: Gaussian Blur-> Sobels Filters -> Non- Maximum Suppression -> Historesis Thresholding
```
lower = 0
upper = 50
edges = cv2.Canny(gray, lower, upper)
```
[Example](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/5.%20Canny%20Edge%20Detection.ipynb)

### 3. Hough Transformation: 
Used to detect line, circles and other shapes 

* [HoughLines Notebook](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_1.%20Hough%20lines.ipynb)

* [HoughCircles Notebook](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_2.%20Hough%20circles%2C%20agriculture.ipynb)

### 4. Cascade Classifier: 

Used to detect object

```
# load in cascade classifier
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector on the grayscale image
faces = face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
```
[Example](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/7.%20Haar%20Cascade%2C%20Face%20Detection.ipynb)

## Types of Feature Image Segmentation:
 
### 1. Harris Corner Detector:

To Detect Corner

```
#Syntax
dst = cv2.cornerHarris(gray_img, blockSize, kSize, k)
```
[Example](https://github.com/udacity/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/1.%20Harris%20Corner%20Detection.ipynb)

### 2. Find Countour:

To findContours

```
# Find contours from thresholded, binary image
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```
[Example](https://github.com/udacity/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/2.%20Contour%20detection%20and%20features.ipynb)

### 3. K-Mean Clustering:

Devide image based on clusters

```
# define stopping criteria
# you can change the number of max iterations for faster convergence!
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

#select Value of K
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
```
[Example](https://github.com/udacity/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/3.%20K-means.ipynb)
## Object Detection

### ORBs Algorithm(Oriented Fast and Rotated Brief):
Fast- Feature Detection
Brief- Vector Creation
Used to identify Feature Vector in an Image which is used to detect object

#### Fast (Features From Accelerated Segments Test):
Used to get Keypoints
#### BRIEF (Binary Robust Independent Elementary Features):
Used to get Binary Vector From Keypoint

Main Property of ORBs:
* Scale Invariance
* Rotational Invariance
* Illumination Invariance
* Noise Invariance

checkout this Awesome [Example](https://github.com/udacity/CVND_Exercises/blob/master/1_4_Feature_Vectors/2.%20ORB.ipynb) for more details

Quick Note: 
How to Downsample Image?
```
#downsample 1/2 of image
level_1 = cv2.pyrDown(image)
#downsample 1/4 of image
level_2 = cv2.pyrDown(level_1)
#downsample 1/8 of image
level_3 = cv2.pyrDown(level_2)
#and so on...
```

Applications: 
* To detect Object in realtime video streaming.
* It Works well for face detection

Limitation: 
*  It can not be used for general object detection like pedestrian detection because of varing clothing and other conditions

### HOGs (Histogram of Oriented Gradient): 
Eliminates the above limitations

[Notebook of HOGs Concept](https://github.com/udacity/CVND_Exercises/blob/master/1_4_Feature_Vectors/3_1.%20HOG.ipynb)

[Example](https://github.com/udacity/CVND_Exercises/blob/master/1_4_Feature_Vectors/3_2.%20HOG%20Examples.ipynb)

## CNN

#### What is batch Normalization?
The idea being that, instead of just normalising the inputs to the network, we normalise the inputs to layers within the network. It’s called “batch” normalization because during training, we normalise the activations of the previous layer for each batch, i.e. apply a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
#### Benefits:
* Networks train faster
* Allows Higher Learning rate
* Makes weights more easiler to intialize
* Makes more Activation function viable
* Simplifies the creation of deep networks
* Provides more Regulation

[Beautiful Article on Batch Normalization](https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82##targetText=Using%20batch%20normalisation%20allows%20much,to%20the%20initial%20starting%20weights)

### SmoothL1Loss: 
It is used for box regression on some object detection system

### Adam algorithm : ###
Here are some of the benefits of the Adam algorithm:

* Straightforward to implement
* Computationally efficient
* Little memory requirements
* invariant to diagonal rescale of the gradients
* Well suited for problems that are large in terms of data and/or parameters
* Appropriate for non-stationary objectives
* Appropriate for problems with very noisy/or sparse gradients
* Hyper-parameters have intuitive interpretation and typically require little tuning

## Advanced CNN ##

### Localization in the Image : ###
Finding location of each object in the image

### Regression Losses ###
* L1Loss: Distance b/w Predicted and True Points, Negligible for small error value

* MSELoss: Mean Square Error b/w object of Predicted and True points, emplifies big error
MSELoss is calculated by avg(sqare(t-p)) points

* SmoothL1Loss: For Small Error uses MSELoss and for High loss uses L1Loss

## Region Proposal : ##
generate regions only for the area that might be object

#### R-CNN : #### 
R-CNN are used in Tracking(Detecting and Recognizing) Object in Motion 
process each region individually
Slow

#### Fast R-CNN ####
Process image only once through convoution layer and then selectively process other region
ROIPooling
Train fast compare to RCNN
Slower for test images
still looks at unwanted images

#### FasterR-CNN ####
Idea is to decrease time to form regional proposal
Has seperate Regional Proposal Network for this purpose
This Regions goes under binary classification of object/Non Object
Fastest compare to earlier networks


[Article on RCNN](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)


YOLO(You Only Look Once)
IoU(Intersection over Union): Inetersection/Union of Regions (range(0,1))

Non Maximal Suppression: 
Only select Regions with high Pc value
used to minimize no of regions

Anchor Boxes:
To select multiple objects having same grid cell as a center point

Temporal Dependencies: Dependency over time

### FFNN ###
Its Simple NN
FeedForward Neural Network are unable to detect Temporal Dependencies

### RNN ### 
Not Only Current Input but also past input

all the simple RNN Suffors Vanishing Gradient Problem in which contribution of information decreases geometrically over time.

### Exploding Gradient ###
Simple Solution to the explooding Gradient Problem is gradient clipping
