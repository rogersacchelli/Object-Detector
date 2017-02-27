# Object Detection

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[test1_original]: ./output_images/test1.jpg
[test1_y]: ./output_images/ycrcb_0.jpg
[test1_cr]: ./output_images/ycrcb_1.jpg
[test1_cb]: ./output_images/ycrcb_2.jpg
[hog_y]: ./output_images/hog_0.jpg
[hog_cr]: ./output_images/hog_cr.jpg
[hog_cb]: ./output_images/hog_cb.jpg
[sliding_win]: ./output_images/window_scaling.jpg
[heat_map]: ./output_images/heat_map.png

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

Y Hog|Cr Hog|Cb Hog
----|----|----
![][hog_y]|![][hog_cr]|![][hog_cb]

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
	
### Color Space transform

Each feature extracted was extracted not from default RGB image but from an transformed image to YCrCb color space.

| Original Image | Y image|Cr Image|Cb Image|
|----|----|----|----|
|![][test1_original]|![][test1_y]|![][test1_cr]|![][test1_cb]|

***Find in the code:***

* image_classifier:
	* extract_features():74:86
	
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

The classifier has been trained using Support Vector Machine algorithm, with default parameters.

Car objetcs are classified with class equal zero, while non-car objects are classified as 1.

It achieved and accuracy of 0.99% with a test size of 10% of augmented dataset.

***Find in the code:***

* fit_svm()
	* image_classifier.py:168:200

![][image1]

### Sliding Window for classification input

In order to correctly perform the image classification, the input image must have the same size of the image previously trained, 64x64 pixels. As we can not simply input the whole image to the classifier, the solution is to patch the whole image into multiple windows of 64x64 pixels.

Image is first croped to the region of interest which is roughly the half bottom of the image, which also helps to reduce the number of false positives.

![][sliding_win]

Each observed square is the result of a 50% overlap between the windows, on both x and y axis. Although it seems that the evaluated patch is the observed small square, the evaluated patch is a 64x64 picture. 

**Window Scaling**

In order to better identify large figures as it's close to evaluation perspective, an scaling factor can be added to increase window size. The scaling factor consists of a list of factor which is intented do evaluated, for example:

> SCALING_FACTOR = [1,2] ----> win_size = [64,128]

***Find in the code***

* object_detector:
	* get_objects_fast():102:141

### Heat Map for false positive removal
If the pipeline included only the output from the classifier, we would see considerable false posivite. Heat Map function is included to discard detected pixels with low number of detected counting.

![heat_map][heat_map]

### Final Result
