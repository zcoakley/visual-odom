import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Set ground truth filepath
ground_truth_file = "ground_truth.txt"
FRAME_0_ID = "000000"
FRAME_1_ID = "000001"

NUM_IMAGES = 2760
MIN_FEATURES = 2000
MAX_FRAME = 1000 # only use the first n frames

# Load image as grayscale
img0 = cv2.imread(f'data/image_0/{FRAME_0_ID}.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(f'data/image_0/{FRAME_1_ID}.png', cv2.IMREAD_GRAYSCALE)

if img0 is None or img1 is None:
    print("Error: Could not load images!")
    print("Current working directory:", os.getcwd())
    exit()

# Display an image
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Camera calibration constants
FOCAL = 718.8560 # value from the KITTI dataset that says how zoomed in the camera is
PP = (607.1928, 185.2157) # value from the KITTI dataset indicating where the center of the camera is


def detect_features(img):
    # FAST feature detector

    # find and draw the keypoints
    keypoints = fast.detect(img,None)
    visualization_img = cv2.drawKeypoints(img, keypoints, None, color=(255,0,0))

    # Print all default params
    # print( "Threshold: {}".format(fast.getThreshold()) )
    # print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    # print( "neighborhood: {}".format(fast.getType()) )
    # print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

    # save image of features
    cv2.imwrite('features.png', visualization_img)

    # convert keypoints to numpy array
    keypoints = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    return keypoints


def track_features(old_image, new_image, old_keypoints):
    # Feature tracking using KLT aka LK Optical Flow

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

    # Convert a copy of the KeyPoint objects to the format needed for optical flow
    # old_keypoints_copy = np.array([kp.pt for kp in old_keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # calculate optical flow
    new_keypoints, status, err = cv2.calcOpticalFlowPyrLK(old_image, new_image, old_keypoints, None, **lk_params)
    status = status.reshape(-1)

    # Filter to only include successfully tracked points
    old_keypoints = old_keypoints[status==1]
    new_keypoints = new_keypoints[status==1]

    # # Create a mask for drawing
    # mask = np.zeros_like(new_image)
    # # Create random colors for each feature point
    # color = np.random.randint(0, 255, (len(old_keypoints), 3))

    # # Draw the tracks
    # for i, (new, old) in enumerate(zip(new_keypoints, old_keypoints)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     # Draw line showing movement
    #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    #     # Draw circle at new position
    #     new_image = cv2.circle(new_image, (int(a), int(b)), 5, color[i].tolist(), -1)

    # # Combine the image with the motion lines
    # result = cv2.add(new_image, mask)

    # # Display the result
    # cv2.imshow('Feature Tracking', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Save the result
    # cv2.imwrite('optical_flow_result.jpg', result)

    return old_keypoints, new_keypoints


def calculate_essential_matrix(points1, points2):
    # Make sure keypoints have the correct shape
    points1 = np.array(points1, dtype=np.float32).reshape(-1, 2)
    points2 = np.array(points2, dtype=np.float32).reshape(-1, 2)
    
    E, mask = cv2.findEssentialMat(
        points1, points2,
        FOCAL,
        PP,
        cv2.RANSAC,
        0.999,
        1.0
    )

    # Use only the points that are inliers of the essential matrix
    inliers1 = points1[mask.ravel() == 1].reshape(-1, 1, 2)
    inliers2 = points2[mask.ravel() == 1].reshape(-1, 1, 2)

    return E, inliers1, inliers2


def recover_pose(E, points1, points2):
    retval, R, t, mask = cv2.recoverPose(E, points1, points2, focal=FOCAL, pp=PP)
    return R, t


def get_absolute_scale(frame_id, ground_truth_filepath):
    """
    Read ground truth poses to get absolute scale.
    
    Normally this would be done using some external sensor data. frame_id should be the 
    name of the second frame.
    """
    try:
        with open(ground_truth_filepath, 'r') as f:
            lines = f.readlines()
        
        if frame_id >= len(lines) or frame_id < 1:
            return 0
        
        # Parse current and previous frame poses
        curr_pose = np.array([float(x) for x in lines[frame_id].split()])
        prev_pose = np.array([float(x) for x in lines[frame_id - 1].split()])
        
        # Extract translation - (positions 3, 7, 11 are x, y, z)
        curr_t = np.array([curr_pose[3], curr_pose[7], curr_pose[11]])
        prev_t = np.array([prev_pose[3], prev_pose[7], prev_pose[11]])
        
        # Return Euclidean distance
        return np.linalg.norm(curr_t - prev_t)
    except Exception as e:
        print(f"Error reading ground truth: {e}")
        return 0
    
def load_ground_truth_poses(file):
    """Load ground truth poses (T_w_cam0) from file."""
    pose_file = file

    # Read and parse the poses
    poses = []
    with open(pose_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            poses.append(T_w_cam0)

    return np.array(poses)


# Initiate FAST feature detector object
fast = cv2.FastFeatureDetector_create(threshold=20)

keypoints_0 = detect_features(img0)
keypoints_0, keypoints_1 = track_features(img0, img1, keypoints_0)

essential_matrix, keypoints_0, keypoints_1 = calculate_essential_matrix(keypoints_0, keypoints_1)
# R is a 3x3 matrix representing which way the robot is facing with 6DOF (x, y, z, roll, pitch, yaw)
R, t = recover_pose(essential_matrix, keypoints_0, keypoints_1)

# print(t)

# Initialize trajectory
R_f = R.copy()
t_f = t.copy()

# Ground truth poses
ground_truth_poses = load_ground_truth_poses(ground_truth_file)

# Create trajectory visualization window
traj = np.zeros((600, 600, 3), dtype=np.uint8)

prev_image = img1
prev_features = keypoints_1

# Create windows
cv2.namedWindow('Road facing camera', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Trajectory', cv2.WINDOW_AUTOSIZE)

print("Starting visual odometry...")


# Loop through all images to calculate odom movements
for image_num in range(2, min(MAX_FRAME, NUM_IMAGES)):
    # Read current frame
    curr_image = cv2.imread(f'data/image_0/{image_num:06d}.png', cv2.IMREAD_GRAYSCALE)
    # Color version for display
    curr_image_color = cv2.imread(f'data/image_0/{image_num:06d}.png')

    if curr_image is None:
        print(f"Error: Could not load image {image_num:06d}!")
        print("Current working directory:", os.getcwd())
        exit()
    
    # Track features
    prev_features, curr_features = track_features(
        prev_image, curr_image, prev_features
    )

    # Mystery feature conversion
    # prev_features = np.array([kp.pt for kp in keypoints_0], dtype=np.float32).reshape(-1, 1, 2)
    
    # Need at least 5 features for 5-point algorithm
    if len(prev_features) < 5:
        print(f"Frame {image_num:06d}: Too few features to continue")
        break
    
    # Compute essential matrix and recover pose
    essential_matrix, prev_features, curr_features = calculate_essential_matrix(prev_features, curr_features)
    R, t = recover_pose(essential_matrix, prev_features, curr_features)
    
    # Get absolute scale from ground truth
    scale = get_absolute_scale(image_num, ground_truth_file)

    print(f"Scale {scale}")
    
    # Update pose if scale is reasonable and forward motion dominates
    if scale > 0.05 and abs(t[2]) > abs(t[0]) and abs(t[2]) > abs(t[1]):
        t_f = t_f + scale * (R_f @ t)
        R_f = R @ R_f
    else:
        print(f"Filtered out from scale {scale}")

    
    # Re-detect features if too few
    if len(prev_features) < MIN_FEATURES:
        prev_features = detect_features(prev_image)
        curr_features, status = track_features(
            prev_image, curr_image, prev_features
        )
    
    # Update for next iteration
    prev_image = curr_image.copy()
    prev_features = curr_features

    # print(t_f)




    # ------------VISUALIZATION----------------
    t_flat = t_f.flatten()

    # Draw trajectory (bird's eye view)
    # Convert world coordinates to pixel coordinates for display
    drawing_scale = 1 # set scale to size the trajectory to fit into the display box
    x = int(t_flat[0] * drawing_scale) + 200
    y = -1 * int(t_flat[2] * drawing_scale) + 200  # Use z-coordinate for forward motion

    # print(type(y))
    # print(ground_truth_poses)


    # Draw the predicted path and ground truth
    cv2.circle(traj, (x, y), 1, (0, 0, 255), 2)

    gt_scale = 1 # set display scale for ground truth data
    gt_x = int(ground_truth_poses[image_num][0][3] * gt_scale) + 200
    gt_y = int(ground_truth_poses[image_num][2][3] * gt_scale) + 200
    cv2.circle(traj, (gt_x, gt_y), 1, (0, 255, 0), 2)

    print("GT raw not scaled:", ground_truth_poses[image_num][0][2],
        ground_truth_poses[image_num][2][2])
    print("GT predicted not scaled:", t_flat[0], t_flat[2])
    print("GT predicted:", x, y)
    print("GT raw:", gt_x, gt_y)

    # Draw info box
    cv2.rectangle(traj, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED)
    text = f"Coordinates: x={t_flat[0]:.2f}m y={t_flat[1]:.2f}m z={t_flat[2]:.2f}m"
    cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 
            1, (255, 255, 255), 1, 8)

    # Show current frame with tracked features
    display_img = curr_image_color.copy()
    # print(curr_features[0])
    for pt in curr_features:
        # print(pt)
        # print(type(int(pt[0])))
        cv2.circle(display_img, (int(pt[0][0]), int(pt[0][1])), 3, (0, 255, 0), -1)

    # # Display
    cv2.imshow('Road facing camera', display_img)
    cv2.imshow('Trajectory', traj)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Print progress
    if image_num % 10 == 0:
        # print(t_f[1])
        print(f"Frame {image_num}/{min(MAX_FRAME, NUM_IMAGES)}: "
                f"Features={len(prev_features)}, "
                f"Position=({t_f[0][0]:.2f}, {t_f[1][0]:.2f}, {t_f[2][0]:.2f})")

print("\nVisual Odometry Complete!")
t_flat = t_f.flatten()
print(f"Final position: x={t_flat[0]:.2f}m, y={t_flat[1]:.2f}m, z={t_flat[2]:.2f}m")

# Save final trajectory
cv2.imwrite('trajectory_final.png', traj)
print("Saved trajectory to 'trajectory_final.png'")

cv2.waitKey(0)
cv2.destroyAllWindows()