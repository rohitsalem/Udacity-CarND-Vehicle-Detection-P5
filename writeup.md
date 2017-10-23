
---

# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Train a classifier Linear SVM classifier
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (starting with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_features.png
[image2]: ./output_images/basic_output_test.png
[image3]: ./output_images/basic_heat.png
[image4]: ./output_images/heat_theshold.png
[image5]: ./output_images/heat_labels.png
[image6]: ./output_images/test_out_labels.png
[image7]: ./output_images/window1.png
[image8]: ./output_images/window1_5.png
[image9]: ./output_images/window2.png
[image10]: ./output_images/window3.png
[image11]: ./output_images/test_imgs_out.png
[video1]: ./project_video_out.mp4

# Pipeline
For this pipline I have used the Histogram of Gradients features and omitted the spatial and color histogram techniques,
## Histogram of Oriented Gradients (HOG) 

The part of the code which extracts the HOG features can be found in the cell 3 of the ipython notebook, `P5.ipynb`. The function calculates the hog features given the `image`,  `orient`, `pixel_per_cell`, `cell_per_block` and returns the HOG features which are used as an input to the classifer in further steps. 
Here is an image which shows an example images for both car and non-car, after performing the HOG features: 
![alt text][image1]

## Processing the data and training the Classifier:

* The `extract_features` function performs a hog operation and returns the features in the form of an array when we have an array of images, can be found in cell 5. 

* The data (both the car and nor-car images) is passed into the `extract_features` function and once we have all the hog features as an array we can then split them as train and test data using the sklearn `train_test_split` function. The labels assigined were 1 for car and 0 for non-car images. 

* As recommended in the lesson, I used the SVM linear classifier and trained the model with the features obtained from above, Obtained an train accuracy of 98% for both train and test data, which validated the model as well. 

## Detecting the cars in an image:

* We can now detect the cars but only if the major portion of the input image is car, which is not a case here, So we need a sliding window which covers the image and then the classifier can be run on that windows one at a time. As of now I also restricted the search space to the bottom half of the image, where the cars are ought to be found. 
* The code for finding cars and drawing boxes can be found in cells 8, 9 and 10. 
The output for the first run is here:

![alt text][image2]

## Using different sizes of windows for the search:

* As the cars can be found in changing scales when the cars move with respect to the camera mounted, I searched for various scales for the cars in images along with varying the `ystop`.
The various scales can be seen in these images:
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

## Using Heatmap

* Created a heatmap by increasing the intensity of a pixel when the car is found multiple times, then thresholding the heatmap to avoid false positives which are most likely detected in one or two windows. 

The raw heatmap :

![alt text][image3]

The heatmap after thresholding:

![alt text][image4]

* Then I label the heatmaps to fing the actual number of cars using Scipy labels
 The labeled heatmap: 
 ![alt text][image5]
 
 * Then I find the max and min corners covering the labels, so that we can have a clear bounding box covering the car totally.
 
 ![alt text][image6]

* The Whole pipeline run once on test images can be found in the image below:
![alt text][image11]

## Running on the Videos:

* First I ran the pipeline directly on the test video, predicting frame to frame without taking into consideration the rectangles detected in the previous frames, it can be found in [video](./test_out1.mp4)
* To take into consideration, previously detected rectangles, `Detect_vehicle` class is used to append rectangles from the previous frames while discarding the oldest frames. The pipeline using this can be found in the test [video](./test_out_2.mp4)
* Finally I ran the Pipeline, considering the previously detected rectangles on the main video. The results can be found [here](./project_video_out.mp4)

## Discussion

* In the output video for the project, the pipeline was able to detect the cars correctly and there were no false positives detected at all which is a good sign, thanks to the heat_map thersholding. 
* The Bounding Boxes detected were sometimes smaller than expected, I think these can be reduced by fine tuning of the scale of sliding windows used for the detection. 
* The Classifier used here is an SVM classifier which is one of the basic classifiers, I think the pipeline can be better handeled by the deep neural networks and there are evident examples in the past like the [F-RCNNs](https://arxiv.org/abs/1506.01497) (Faster Region based Covolutional Neural Networks) which work basically the same way of sliding windows, but are proved much more accurate and fast. 
* Other Object detection algorithms like [YOLO](https://pjreddie.com/darknet/yolo/) and [SSD](https://arxiv.org/pdf/1512.02325.pdf) also seem very promising. But there are a lot of false postives in the output with these mostly because they are trained on various classes which will sometimes confuse the classifier. 
* I would like to improve the pipeline by using Deep neural networks and the other techniques mentioned above along with the HOG features as input in addition to the image itself.
