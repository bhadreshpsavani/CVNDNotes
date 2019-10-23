# My Notes Regarding Machine Vision 

## Convolution Filter Edge Detection: 

### [Fouriour Transform](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/1.%20Fourier%20Transform.ipynb):
* Fouriour Transformation is used in CV to transform Image(Spatial) domain to freq domain
* it can help to visualize changes(Freq) in the images 

### [Canny Edge Detector](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/5.%20Canny%20Edge%20Detection.ipynb): 
Widely used, accurate edge detection algorithm
internally: Gaussian Blur-> Sobels Filters -> Non- Maximum Suppression -> Historesis Thresholding
```
lower = 0
upper = 50
edges = cv2.Canny(gray, lower, upper)
```

### Hough Transformation: 
Used to detect line, circles and other shapes 

*[HoughLines](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_1.%20Hough%20lines.ipynb)

*[HoughCircles](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_2.%20Hough%20circles%2C%20agriculture.ipynb)

### [Cascade Classifier](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/7.%20Haar%20Cascade%2C%20Face%20Detection.ipynb): 

Used to detect object

```
# load in cascade classifier
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector on the grayscale image
faces = face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
```

## Types of Feature Image Segmentation:
 
### [Harris Corner Detector](https://github.com/udacity/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/1.%20Harris%20Corner%20Detection.ipynb):

To Detect Corner

```
#Syntax
dst = cv2.cornerHarris(gray_img, blockSize, kSize, k)
```

### [Find Countour](https://github.com/udacity/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/2.%20Contour%20detection%20and%20features.ipynb):

To findContours

```
# Find contours from thresholded, binary image
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

### [K-Mean Clustering](https://github.com/udacity/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/3.%20K-means.ipynb):

Devide image based on clusters

```
# define stopping criteria
# you can change the number of max iterations for faster convergence!
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

#select Value of K
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
```

## Object Detection

### ORBs Algorithm(Oriented Fast and Rotated Brief):
Fast- Feature Detection
Brief- Vector Creation
Used to identify Feature Vector in an Image which is used to detect object

#### Fast (Features From Accelerated Segments Test):
Used to get Keypoints
#### BRIEF (Binary Robust Independent Elementary Features):
Used to get Binary Vector From Keypoint

* ORBs is Scale and Rotational Invarient

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
*  It can not be used for general object detection like pedestrian because of varing clothing and other conditions