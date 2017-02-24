import numpy as np
import cv2
import os
import pickle
import time
from glob import glob
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


MODEL_SIZE = 100000
COLOR_SPACE = 'RGB'

def get_hist(img, nbins=32, bins_range=(0, 256)):

    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


def get_hog(img, channel, ornt=9, pxl_per_cell=(8,8), cells_per_blk=(2,2), feature_vector=True):

    features = hog(img[:, :, channel],
                   orientations=ornt,
                   pixels_per_cell=pxl_per_cell,
                   cells_per_block=cells_per_blk,
                   transform_sqrt=True,
                   feature_vector=feature_vector)

    return features


def get_spatial(img, size=(32, 32)):

    """
    While it could be cumbersome to include three color
    channels of a full resolution image, you can perform
    spatial binning on an image and still retain enough
    information to help in finding vehicles.

    Even going all the way down to 32 x 32 pixel
    resolution, the car itself is still clearly
    identifiable by eye, and this means that the
    relevant features are still preserved at this
    resolution.
    """
    return cv2.resize(img, size).ravel()

# Image Classifier for object detection


def extract_features(img, hist_feat=True, spatial_feat=True, hog_feat=True, pre_hog=None, color_space=COLOR_SPACE):

    """
    Reads an image and return it features.
    :param img: image path or image itself
    :param hist_feat: if set, computes histogram
    :param spatial_feat: if set, turn image into 32x32 spatial set and insert it to feature set
    :param hog_feat: computes histogram of oriented gradients
    :param color_space: [RGB, HSV, LUV, LAB, YUV, YcrCb]
    :return: image feature map (1D array)
    """

    image_feature = []
    if type(img) is str:
        if color_space == 'RGB':
            img_clr_spc = cv2.imread(img, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            img_clr_spc = cv2.imread(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            img_clr_spc = cv2.imread(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'LAB':
            img_clr_spc = cv2.imread(img, cv2.COLOR_BGR2LAB)
        elif color_space == 'YUV':
            img_clr_spc = cv2.imread(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            img_clr_spc = cv2.imread(img, cv2.COLOR_BGR2YCrCb)
    else:
        if color_space == 'RGB':
            #img_clr_spc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_clr_spc = img
        elif color_space == 'HSV':
            img_clr_spc = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img_clr_spc = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'LAB':
            img_clr_spc = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif color_space == 'YUV':
            img_clr_spc = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            img_clr_spc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    if img_clr_spc.shape[:2] != [64, 64]:
        img_clr_spc = cv2.resize(img_clr_spc, (64, 64))

    if spatial_feat:
        image_feature.append(get_spatial(img_clr_spc))

    if hist_feat:
        image_feature.append(get_hist(img_clr_spc))

    if hog_feat:
        image_feature.append(get_hog(img_clr_spc, channel=0))
        image_feature.append(get_hog(img_clr_spc, channel=1))
        image_feature.append(get_hog(img_clr_spc, channel=2))
    else:
        image_feature.append(pre_hog)

    return np.concatenate(image_feature)


def create_dataset(dataset_size=MODEL_SIZE):

    if not os.path.exists('dataset.p'):
        car_dirs = glob('vehicles/*/*.png')
        non_car_dirs = glob('non-vehicles/*/*.png')

        car_feature_map = []
        for car in car_dirs[0:dataset_size]:
            # from each image from dataset, add more 3 images from data augmentation
            # operations: blur, histogram equalization, flip
            # flip
            #car_feature_map.append(extract_features(cv2.flip(cv2.imread(car),flipCode=1),color_space='RGB'))
            # blur
            #car_feature_map.append(extract_features(cv2.blur(cv2.imread(car, cv2.COLOR_BGR2RGB), (5,5)), color_space='RGB'))
            # histogram_equalization
            #car_yuv = cv2.cvtColor(cv2.imread(car), cv2.COLOR_BGR2YUV)
            #car_yuv[:, :, 0] = cv2.equalizeHist(car_yuv[:, :, 0])
            #car_feature_map.append(extract_features(cv2.cvtColor(car_yuv, cv2.COLOR_YUV2RGB), color_space='RGB'))
            # original image
            car_feature_map.append(extract_features(car, color_space=COLOR_SPACE))
        car_feature_map = np.concatenate([car_feature_map])
        car_label = np.zeros(shape=(car_feature_map.shape[0]), dtype=int)

        non_car_feature_map = []
        for non_car in non_car_dirs[0:dataset_size]:
            # from each image from dataset, add more 3 images from data augmentation
            # operations: blur, histogram equalization, flip
            # flip
            #non_car_feature_map.append(extract_features(cv2.flip(cv2.imread(non_car), flipCode=1), color_space='RGB'))
            # blur
            #non_car_feature_map.append(extract_features(cv2.blur(cv2.imread(non_car, cv2.COLOR_BGR2RGB), (5,5)), color_space='RGB'))
            # histogram_equalization
            #non_car_yuv = cv2.cvtColor(cv2.imread(non_car), cv2.COLOR_BGR2YUV)
            #non_car_yuv[:, :, 0] = cv2.equalizeHist(non_car_yuv[:, :, 0])
            #non_car_feature_map.append(extract_features(cv2.cvtColor(non_car_yuv, cv2.COLOR_YUV2BGR), color_space='RGB'))
            # original image
            non_car_feature_map.append(extract_features(non_car, color_space=COLOR_SPACE))
        non_car_feature_map = np.concatenate([non_car_feature_map])
        non_car_label = np.ones(shape=(non_car_feature_map.shape[0]), dtype=int)

        train_data = np.vstack((car_feature_map, non_car_feature_map)).astype(np.float64)
        label_data = np.hstack((car_label, non_car_label))

        dataset = [train_data, label_data]

        with open('dataset.p', mode='wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    else:
        with open('dataset.p', mode='rb') as f:
            dataset = pickle.load(f)
            f.close()

    return dataset


def fit_svm(dataset):

    X = dataset[0]
    y = dataset[1]

    X_norm = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_norm.transform(X)
    scaled_X, y = shuffle(scaled_X, y)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1)

    print('Starting SVC Training \n'
          'Train Size = %d \n'
          'Test Size = %d' % (len(X_train), len(X_test)))

    clf_t0 = time.time()
    clf_svm = LinearSVC(C=1,verbose=1)
    #clf_sgd = SGDClassifier(learning_rate='optimal')
    clf = clf_svm
    clf.fit(X_train, y_train)
    print('SVC Classification Time: %f' % round(time.time() - clf_t0, 3))

    svm_clf_scaler = [clf, X_norm]

    with open('svm_clf_scaler.p', mode='wb') as f:
        pickle.dump(svm_clf_scaler,f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    pred = clf.predict(X_test)

    print('Test Accuracy: %f' % clf.score(X_test, y_test))
    print('Recall: %f' % round(recall_score(y_test, pred), 3))
    print('F1 Score: %f' % round(f1_score(y_test, pred), 3))

    return svm_clf_scaler


def get_classifier():
    if os.path.exists('svm_clf_scaler.p'):
        with open('svm_clf_scaler.p', mode='rb') as f:
            svm_clf_scaler = pickle.load(f)
            f.close()
        return svm_clf_scaler
    else:
        return fit_svm(create_dataset(MODEL_SIZE))