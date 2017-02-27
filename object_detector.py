import imageio
from image_classifier import *
from scipy.ndimage.measurements import label
from frame_to_video import *

DEFAULT_IMAGE_SIZE = (720, 1280)
IMAGE_ROI = (.0, .55)
SCALING_FACTOR = [1, 1.5, 2]


WIN_SIZE = 64
OVERLAP_FACTOR = 0.5


def get_windows(image_shape=DEFAULT_IMAGE_SIZE,
                xy_window=(64, 64),
                xy_overlap=(OVERLAP_FACTOR, OVERLAP_FACTOR),
                xy_start_pos=IMAGE_ROI,
                scaling_factor=SCALING_FACTOR):

    """
    Slides through input images and returns the rectangles for image slicing
    :param image_shape: Shape size to calculate windows
    :param xy_window: Window size in pixels
    :param xy_overlap: overlap percentage for image scanning
    :param xy_start_pos: Portion of the image in percentage which window search starts
    :param scaling_factor: Determines the scaling factor of adjustment for window
    :return:
    """
    # If x and/or y start/stop positions not defined, set to image size

    # Initialize a list to append window positions to
    window_list = []
    for sf in scaling_factor:
        xy_window_scaled = (xy_window[0] * sf, xy_window[0] * sf)
        n_windown_x = int(round(((image_shape[1] / xy_window_scaled[0]) - xy_overlap[0]) / xy_overlap[0], 0))
        n_windown_y = int(round(((image_shape[0] / xy_window_scaled[1]) - xy_overlap[1]) / xy_overlap[1], 0))
        for y in range(int(round(xy_start_pos[1] * n_windown_y,0)), n_windown_y):

            y_start_stop = [y * xy_window_scaled[1] * xy_overlap[1],
                             y * xy_window_scaled[1] * xy_overlap[1] + xy_window_scaled[1]]

            for x in range(int(round(xy_start_pos[0] * n_windown_x,0)), n_windown_x):
                x_start_stop = [x * xy_window_scaled[0] * xy_overlap[0],
                                x * xy_window_scaled[0] * xy_overlap[0] + xy_window_scaled[0]]

                window_list.append([(int(x_start_stop[0]), int(y_start_stop[0])),
                                    (int(x_start_stop[1]), int(y_start_stop[1]))])
    return window_list


def predict(img, clf, scaler):
    # load test images to test prediction
    image_feature = scaler.transform(extract_features(img).reshape(1, -1))
    return clf.predict(image_feature)[0]

def get_objects(img, windows, roi=IMAGE_ROI):

    objects = []
    for w in windows:
        img_window = img[w[0][1]:w[1][1], w[0][0]:w[1][0], :]
        #cv2.imshow('window', img_window)
        if img_window.size:
            prediction = predict(img_window, clf, scaler)
            if prediction == 0:
                objects.append(w)

    return np.array(objects)


def get_objects_fast(img, scale=SCALING_FACTOR, cells_per_step=2,
                     pixels_per_cell=8):

    y_start = int(IMAGE_ROI[1] * img.shape[0])
    x_start = int((IMAGE_ROI[0] * img.shape[1]))

    y_end = img.shape[0]
    x_end = img.shape[1]

    boxes = []
    for s in scale:
        roi_image = img[y_start:y_end, x_start:x_end, :]
        resized_img = cv2.resize(roi_image, (int(roi_image.shape[1] / s), int(roi_image.shape[0] / s)))

        if COLOR_SPACE == 'RGB':
            img_clr_spc = resized_img
        elif COLOR_SPACE == 'HSV':
            img_clr_spc = cv2.cvtColor(cv2.imread(resized_img), cv2.COLOR_RGB2HSV)
        elif COLOR_SPACE == 'LUV':
            img_clr_spc = cv2.cvtColor(cv2.imread(resized_img), cv2.COLOR_RGB2LUV)
        elif COLOR_SPACE == 'LAB':
            img_clr_spc = cv2.cvtColor(cv2.imread(resized_img), cv2.COLOR_RGB2LAB)
        elif COLOR_SPACE == 'YUV':
            img_clr_spc = cv2.cvtColor(cv2.imread(resized_img), cv2.COLOR_RGB2YUV)
        elif COLOR_SPACE == 'YCrCb':
            img_clr_spc = cv2.cvtColor(resized_img, cv2.COLOR_RGB2YCrCb)

        hog_ch1 = get_hog(img=img_clr_spc, feature_vector=False, channel=0)
        hog_ch2 = get_hog(img=img_clr_spc, feature_vector=False, channel=1)
        hog_ch3 = get_hog(img=img_clr_spc, feature_vector=False, channel=2)

        nblocks_per_window = (WIN_SIZE // pixels_per_cell) - 1

        nx_hog_steps = int((img_clr_spc.shape[1] / WIN_SIZE - OVERLAP_FACTOR) * cells_per_step**2) - 2
        ny_hog_steps = int((img_clr_spc.shape[0] / WIN_SIZE - OVERLAP_FACTOR) * cells_per_step**2) - 2

        for xb in range(nx_hog_steps):
            for yb in range(ny_hog_steps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog_ch1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog_ch2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog_ch3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pixels_per_cell
                ytop = ypos * pixels_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_clr_spc[ytop:ytop + WIN_SIZE, xleft:xleft + WIN_SIZE], (64, 64))

                # Get color features
                spatial_features = get_spatial(subimg)
                hist_features = get_hist(subimg)

                # Scale features and make a prediction
                test_features = scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                test_prediction = clf.predict(test_features)

                if test_prediction == 0:

                    xbox_left = np.int(xleft * s)
                    ytop_draw = np.int(ytop * s)
                    win_draw = np.int(WIN_SIZE * s)
                    boxes.append([(xbox_left, ytop_draw + y_start),
                                  (xbox_left + win_draw, ytop_draw + win_draw + y_start)])

    return boxes

def draw_boxes(img, bboxes, color=(0, 255, 0), thick=1):
    # make a copy of the image
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    copy = np.copy(img)
    for i, b in enumerate(bboxes):
        if i % 1 == 0:
            cv2.rectangle(copy, (b[0][0], b[0][1]), (b[1][0], b[1][1]), color, thick)
    return copy


def add_heat(heatmap, bbox_list, threshold=3):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    border = img.copy()
    fill = img.copy()
    output = img.copy()

    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(fill, bbox[0], bbox[1], (0, 255, 0), -1)
        cv2.rectangle(border, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.addWeighted(fill, 0.8, output, 0.2, 0, output)
        cv2.addWeighted(border, 0.8, output, 0.2, 0, output)
    # Return the image
    return output


clf, scaler = get_classifier()
wins = get_windows()

vid = imageio.get_reader('project_video.mp4', 'ffmpeg')
video_file = frame_to_video('output_video/final_project_2.avi', resolution=DEFAULT_IMAGE_SIZE[::-1])

for i, img in enumerate(vid):

    if img.shape[:2] != (720, 1280):
        img = cv2.resize(img, (720, 1280))
    objects = get_objects_fast(img)
    heatmap = add_heat(np.zeros(shape=DEFAULT_IMAGE_SIZE, dtype=float), objects)
    labels = label(heatmap)
    final_frame_bgr = draw_labeled_bboxes(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), labels)
    video_file.write(final_frame_bgr)
    cv2.imshow('final', final_frame_bgr)
    #cv2.imwrite('output_video/project_output_' + str(i) + '.jpg', , labels))
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video_file.release()