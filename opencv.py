import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
img0 = cv2.imread('data/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('data/image_0/000001.png', cv2.IMREAD_GRAYSCALE)

# Display it
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def detect_features(img):
    # FAST feature detector

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    keypoints = fast.detect(img,None)
    display_img = cv2.drawKeypoints(img, keypoints, None, color=(255,0,0))

    # Print all default params
    # print( "Threshold: {}".format(fast.getThreshold()) )
    # print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    # print( "neighborhood: {}".format(fast.getType()) )
    # print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

    # save image of features
    cv2.imwrite('features.png', display_img)

    return keypoints


def track_features(old_image, new_image, old_keypoints):
    # Feature tracking using KLT aka LK Optical Flow

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Convert KeyPoint objects to the format needed for optical flow
    old_keypoints = np.array([kp.pt for kp in old_keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # calculate optical flow
    new_keypoints, status, err = cv2.calcOpticalFlowPyrLK(old_image, new_image, old_keypoints, None, **lk_params)

    # Select good points
    good_new = new_keypoints[status==1]
    good_old = old_keypoints[status==1]

    # Create a mask for drawing
    mask = np.zeros_like(new_image)
    # Create random colors for each feature point
    color = np.random.randint(0, 255, (len(good_old), 3))

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Draw line showing movement
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        # Draw circle at new position
        new_image = cv2.circle(new_image, (int(a), int(b)), 5, color[i].tolist(), -1)

    # Combine the image with the motion lines
    result = cv2.add(new_image, mask)

    # Display the result
    cv2.imshow('Feature Tracking', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    cv2.imwrite('optical_flow_result.jpg', result)

    return new_keypoints



keypoints_0 = detect_features(img0)
keypoints_1 = track_features(img0, img1, keypoints_0)
