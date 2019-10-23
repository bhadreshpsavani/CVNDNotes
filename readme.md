## [Fouriour Transform](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/1.%20Fourier%20Transform.ipynb):
* Fouriour Transformation is used in CV to transform Image(Spatial) domain to freq domain
* it can help to visualize changes(Freq) in the images 

## [Canny Edge Detector](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/5.%20Canny%20Edge%20Detection.ipynb): 
widely used, accurate edge detection algorithm
    internally: Gaussian Blur-> Sobels Filters -> Non- Maximum Suppression -> Historesis Thresholding
```
lower = 0
upper = 50
edges = cv2.Canny(gray, lower, upper)
```

## Hough Transformation: 
Used to detect line, circles and other shapes 

[HoughLines](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_1.%20Hough%20lines.ipynb)
[HoughCircles](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_2.%20Hough%20circles%2C%20agriculture.ipynb)

## [Cascade Classifier](https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/7.%20Haar%20Cascade%2C%20Face%20Detection.ipynb): 

Used to detect object

```
# load in cascade classifier
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
# run the detector on the grayscale image
faces = face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
```