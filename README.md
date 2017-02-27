# Object Detection

[//]: # (Image References)
[image1]: ./examples/car_not_car.png

## Project Goals

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Histogram of Oriented Gradientes

Histogram of oriented gradients is an algorithm which returns feature infomation for pixels grouped into blocks of a specific number of cells.

It's used in this project as part of a set of feature information to determine if a patch of an image contains or not a car on it.

### Spatial Bin & Color Histogram

As mentioned above, HOG features was added to Spatial binning and Color Histogram to better identify if the evaluated image patch is regarded as a car or not.

**Spatial Binning** consists of flattening the image patch into a 1-D array and append it to the other features of the image

**Histogram of Features** consistis of separating for each color, the amount of pixels contained by a specific group of tones, being each group defined by the bin of the histogram.


***Find in the code:***

* HOG: object_detector:98
	* image_classifier:get_hog():31:38
* Spatial Binning: object_detector:124
	* image_classifier:get_spatial():43:57
* Color Histogram: object_detector:125
	* image_classifier:get_hist():22:28
	
### Training Classifier

The dataset used to train the classifier has been aquired from [Kitti Dataset](http://www.cvlibs.net/datasets/kitti/).

#### Data Augmentation

In order to improve the classifier accuracy, the dataset has been augmentade by fake data generated from the original dataset. For each image of the original dataset, three other images were generated.

* 3 new images per original image
	* Image Flip Horizontaly
	* Imager Blur
	* Histogram Equalization
	
***Find in the code:***

* create_dataset:105
	* image_classifier.py106:130
	
#### Classifier

The classifier has been trained using Support Vector Machine algorothm, with default parameters.

It achieved and accuracy of 0.99% with a test size of 10% of augmented dataset.

***Find in the code:***

* fit_svm()
	* image_classifier.py:168:200

![alt text][image1]
