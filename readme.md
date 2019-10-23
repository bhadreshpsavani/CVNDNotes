Date: 19-10-2019
Fouriour Transform:
* Fouriour Transformation is used in CV to transform Image(Spatial) domain to freq domain
* it can help to visualize changes(Freq) in the images 

## Canny Edge Detector: 
widely used, accuarate edge detection algorithm
    internally: Gaussian Blur-> Sobels Filters -> Non- Maximum Suppression -> Historesis Thresholding

https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/5.%20Canny%20Edge%20Detection.ipynb

## Hough Transformation: 
Hough on the edge-detected image is used after canny edge detection to make continues line 
https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_1.%20Hough%20lines.ipynb
https://github.com/udacity/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_2.%20Hough%20circles%2C%20agriculture.ipynb

## Cascade Classifier: Used to detect object

```# load in cascade classifier
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
# run the detector on the grayscale image
faces = face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)```
