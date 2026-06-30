@mainpage Visual Odometry SLAM Package

@mainpage Visual Odometry SLAM Package

# visual_odom

## Authors

**Lukas Grob**  
**Paul Schellenberg**

This project was developed by the authors listed above as part of a visual odometry and SLAM research/development project.

---

A ROS 2 package implementing a visual odometry pipeline for a mobile robot equipped
with a Kinect RGB-D camera.
...

# visual_odom

A ROS 2 package implementing a visual odometry pipeline for a mobile robot equipped
with a Kinect RGB-D camera. The system detects ORB keypoints in incoming RGB frames,
lifts them into 3D using registered depth data, and estimates frame-to-frame motion
via RANSAC Kabsch alignment. Pose estimation is embedded in a
Rao-Blackwellized particle filter: each particle maintains its own landmark map and
an EKF per tracked landmark. The best particle is selected by log-likelihood weight.
The resulting pose is broadcast as a TF transform and published as a
`nav_msgs/Odometry` message.

---

## 1. Overview

### UML Diagram
\image html UML_for_ReadMe.svg


### System Pipeline

```
RGB + Depth frames
      |
      v
ORB keypoint detection + depth filter  [MIN_DEPTH .. MAX_DEPTH]
      |
      v
Frame-to-frame RANSAC + Kabsch  -->  delta pose (translation + rotation)
      |
      v
Per-particle update  (N = n_robot_samples particles)
  |-- Wheel odometry EKF prediction (parralel to that)
  |-- Visible landmark query (FOV check)
  |-- BFMatcher: current frame vs last frame
  |-- Apply delta pose to particle position
  |-- BFMatcher: current frame vs visible landmarks
  |-- Landmark EKF update for matched landmarks
  |-- Log-weight calculation
  |
  v
Best particle selection (argmax normalized weight)
      |
      v
TF broadcast (odom -> base_link) + nav_msgs/Odometry
```

### TF Tree

```
odom
  +-- base_link
        +-- kinect_depth   (from /tf_static in rosbag)
```

The `odom -> base_link` transform is updated on every processed RGB frame.
The `base_link -> kinect_depth` static transform is provided by the rosbag.

### ROS Topics

| Topic | Type | Direction |
|-------|------|-----------|
| RGB_IMAGE_TOPIC | `sensor_msgs/Image` | Subscribed |
| DEPTH_IMAGE_TOPIC | `sensor_msgs/Image` | Subscribed |
| WHEEL_ODOMETRY_TOPIC | `nav_msgs/Odometry` | Subscribed |
| VISUAL_ODOM_MSG_TOPIC | `nav_msgs/Odometry` | Published |
| KEYPOINT_POINTCLOUD_FRAME_TOPIC | `sensor_msgs/PointCloud2` | Published -- landmark map |
| POINTCLOUD_FRAME_TOPIC | `sensor_msgs/PointCloud2` | Published -- full depth frame (not used for the main time) |
| VISION_CONE_TOPIC | `visualization_msgs/Marker` | Published |
| KP_IMAGE_TOPIC | `sensor_msgs/Image` | Published |
| VISUAL_ODOM_PATH_TOPIC | `visualization_msgs/MarkerArray` | Published |

### Dependencies

| Package | Purpose |
|---------|---------|
| `rclpy` | ROS 2 Python client library – node lifecycle, publishers, subscriptions, parameters |
| `rcl_interfaces` | `SetParametersResult` for dynamic parameter callbacks |
| `builtin_interfaces` | ROS 2 built-in message types (timestamps, duration) |
| `std_msgs` | `Header` for PointCloud2 and TF stamped messages |
| `sensor_msgs` | `Image`, `PointCloud2`, `PointField`, `Imu` |
| `sensor_msgs_py` | PointCloud2 creation utilities (`point_cloud2.create_cloud`) |
| `nav_msgs` | `Odometry` – wheel, IMU and visual odometry messages |
| `geometry_msgs` | `TransformStamped`, `Point`, `Pose2D`, `Quaternion` |
| `visualization_msgs` | `Marker`, `MarkerArray` – particle path and vision cone visualisation |
| `tf2_ros` | TF broadcaster, listener and buffer for coordinate frame transforms |
| `cv_bridge` | ROS `Image` ↔ OpenCV `Mat` conversion |
| `python3-opencv` | ORB detection, descriptor computation, BFMatcher, image I/O |
| `python3-numpy` | Numerical arrays, linear algebra (EKF matrices, RANSAC, Kabsch) |
| `python3-scipy` | Quaternion/rotation conversions (`scipy.spatial.transform.Rotation`), KDTree |

---

## 2. Package Structure

```
visual_odom/
+-- visual_odom/
|     +-- constants.py          # Global dataclasses, configuration constants, helper functions
|     +-- landmark.py           # Landmark dataclass, EKF wrapper, and Kabsch estimator
|     +-- ekf_landmark.py       # Extended Kalman Filter for individual landmark position
|     +-- ekf_robot.py          # Extended Kalman Filter for robot pose (wheel odom based)
|     +-- robot.py              # VisualRobotSample: one particle in the particle filter
|     +-- visual_odom_map.py    # VisualOdomMap: persistent landmark map per particle
|     +-- tf_methods.py         # Coordinate frame transformation utilities
|     +-- visual_odom_node.py   # Main ROS 2 node: orchestrates particle filter
+-- launch/
|     +-- visual_odom_launch.py # Launch file: node + rosbag playback + RViz2
+-- config/
      +-- param.yaml            # Runtime parameter configuration
+-- rviz/
      +-- rviz2_config.rviz     # Rviz2 configuration for Autostart
```

### Module Descriptions

**Dataclasses**

| Class | Purpose |
|-------|---------|
| `Coordinate` | 3D position container (x, y, z) in meters. Supports arithmetic operators. |
| `PixelCoordinate` | 2D image pixel (u, v) paired with its raw depth value z. |
| `State` | 2D robot pose (x, y, theta) in meters and radians. Supports arithmetic operators. |
| `Parameters` | Runtime parameter bundle populated at node startup from ROS 2 parameter declarations. |

**Key Constant Groups**

| Group | Examples |
|-------|----------|
| ROS Topics | `RGB_IMAGE_TOPIC`, `DEPTH_IMAGE_TOPIC`, `WHEEL_ODOMETRY_TOPIC` |
| TF Frame IDs | `KINECT_FRAME_ID`, `BASE_LINK_FRAME_ID`, `VISUAL_ODOM_FRAME_ID` |
| Camera Model | `F`, `CU`, `CV`, `MAX_AZIMUTH`, `MAX_ALTITUDE`, `ROT_BK` |
| Depth Filter | `MIN_DEPTH`, `MAX_DEPTH` |
| RANSAC | `RANSAC_EVALUATION_TOLERANCE`, `RANSAC_ITERATION`, `RANSAC_MIN_INLIER_RATIO` |
| Particle Filter | `N_ROBOT_SAMPLES`, `NOISE_INCREMENT_RANSAC_FAILURE` |
| Landmark Management | `MIN_LANDMARK_TRUST`, `INCREASE_TRUST_VALUE`, `DECREASE_TRUST_FACTOR` |
| Noise / EKF | `SIGMA_X_ODOM_WHEEL_Q`, `ERROR_MIN_DEPTH`, `ERROR_QUADRATIC_DEPTH` |

**Helper Functions**

- `normalize_angle(angle)` -- Maps any angle in radians into `[-pi, pi]`
  using `(angle + pi) % (2*pi) - pi`.

---

#### landmark.py

Defines the `Landmark` class representing a single tracked 3D map feature. Each
landmark owns an `ExtendedKalmanFilterLandmark` instance that refines its odom-frame
position over time. Also contains the standalone `kabsch()` function used for 2D
rigid-body alignment in RANSAC.

**`Landmark`**

On construction, pixel coordinates (u, v) and depth z are back-projected into 3D
Kinect-frame coordinates via the pinhole camera model:

```
x = z * (u - CU) / F
y = z * (v - CV) / F
```

The odom-frame position is set separately via `set_odom_coordinates()` after
the coordinate frame transform is applied.

| Method | Description |
|--------|-------------|
| `is_visible(pos_camera_odom, theta_robot)` | Checks azimuth, altitude, and depth constraints against the camera FOV. |
| `landmark_kalman_iteration(pos, theta, pixel_coor)` | Delegates one EKF predict+update cycle to the embedded `ExtendedKalmanFilterLandmark`. |
| `calculate_likelihood(z_pos_odom, theta)` | Computes the likelihood of a measurement given the current landmark state and covariance. Used for particle weight calculation. |
| `increase_trust()` / `decrease_trust()` | Trust management: additive increment on observation, multiplicative decay when not seen. |
| `get_odom_coordinates()` / `set_odom_coordinates(c)` | Accessors for the odom-frame position. |
| `get_descriptor()` / `get_kp()` | Accessors for the ORB descriptor and keypoint. |

**`kabsch(P_i, Q_i)`**

Computes the optimal 2D rotation R and translation t that aligns point set Q onto P
in a least-squares sense. Mean-centers both sets; theta is derived analytically via
`atan2`. Returns `(R, t, theta)` or `(None, None, None)` for degenerate inputs.

---

#### ekf_landmark.py

Implements the Extended Kalman Filter for tracking an individual landmark position
in the odom frame. The state vector is a 3D Coordinate `(x, y, z)`.

**Measurement model**

The expected measurement transforms the landmark from odom into the robot's local
frame using the current heading rotation:

```
h(x) = R(theta).T * (x_landmark - x_robot)
```

**Noise model**

The measurement covariance R is depth-dependent, propagated from pixel-space noise
to baselink:

```
R_baselink = ROT_BK * J_kinect * R_pixel * J_kinect.T * ROT_BK.T
```

where the depth error grows quadratically: `s_z = ERROR_MIN_DEPTH + ERROR_QUADRATIC_DEPTH * (z - z_min)^2`.

| Method | Description |
|--------|-------------|
| `landmark_kalman_iteration(pos, theta, pixel_coor)` | Full predict + update cycle. Returns updated `(x, P)`. |
| `prediction(x)` | Applies identity state transition; propagates covariance with process noise Q. |
| `update(x_tt1, P_tt1, pos_robot, theta, kp)` | Computes Kalman gain, innovation, and Joseph-form covariance update. |
| `computeKalmanGain(P_tt1)` | Returns `K = P * H.T * (H * P * H.T + R)^-1`. |

---

#### ekf_robot.py

Implements the Extended Kalman Filter for the robot pose `(x, y, theta)` using
wheel odometry as the prediction input and RANSAC visual odometry as the measurement.

The state transition is additive: `x_{t|t-1} = x_{t-1} + delta_wheel_odom`.
The measurement noise R scales with the RANSAC evaluation tolerance plus any
accumulated additional noise from repeated RANSAC failures.

| Method | Description |
|--------|-------------|
| `prediction(delta_wheel_odom)` | Propagates state and covariance with wheel odometry delta. |
| `update(z, inlier_ratio)` | Fuses RANSAC pose measurement using Joseph-form covariance update. |
| `add_noise_to_R()` | Increments additional measurement noise after a RANSAC failure, capped at `MAX_ADDITIONAL_NOISE_RANSAC_FAILURE`. |

---

#### robot.py

Represents a single particle in the Rao-Blackwellized particle filter. Each particle
maintains its own `VisualOdomMap`, `ExtendedKalmanFilterRobot`, and accumulated pose.
All particles receive the same RANSAC result computed centrally by `VisualOdom` and
apply it independently.

**`robot_iteration(...)`**

The per-frame update sequence for one particle:

1. On first iteration: seed the landmark map until `INITIAL_LANDMARK_COUNT` is reached.
2. Apply the RANSAC delta pose to the particle position with optional Gaussian noise.
3. Query `get_visible_landmarks()` for landmarks in the current camera FOV.
4. Match visible landmark descriptors against current frame descriptors via BFMatcher.
5. Run `landmark_kalman_iteration()` on all matched landmarks.
6. Compute `log_weight` via `calculate_log_weight()`.
7. Run `cleanup_old_landmarks()`: increase trust for matched landmarks, decay others,
   remove landmarks below `MIN_LANDMARK_TRUST`.
8. Add unmatched keypoints as new landmarks if match count is below
   `MATCHES_FOR_NEW_LANDMARKS`.

**`predict_with_wheel_odom(state_wheel_odom)`**

Computes delta from the last stored wheel odometry state and feeds it to
`ExtendedKalmanFilterRobot.prediction()`.

**`ransac_failed()`**

Falls back to wheel odometry delta for position update and increments noise in
the robot EKF via `add_noise_to_R()`.

---

#### visual_odom_map.py

Implements `VisualOdomMap`, a `list` subclass holding all `Landmark` objects for
one particle. Provides spatial queries, batch EKF updates, log-weight calculation,
and PointCloud2 publishing.

| Method | Description |
|--------|-------------|
| `add_landmark(landmark)` | Appends a single landmark. |
| `add_landmarks_from_kps(P_init, kps, des, frame_rgb, kp_depth, theta, pos_baselink)` | Creates landmarks from ORB keypoints, computes odom-frame coordinates, and add the keypoints as landmarks to the map. |
| `get_visible_landmarks(camera_pos, theta)` | Returns a new `VisualOdomMap` containing only landmarks passing the `is_visible()` FOV check. |
| `get_descriptors()` | Returns all descriptors stacked as a NumPy array for batch BFMatcher input. |
| `landmark_kalman_iteration(pos, theta, indices, kp_pos)` | Runs one EKF iteration on each indexed landmark with its matched pixel coordinate. |
| `calculate_log_weight(pos, theta, indices, kp_pos)` | Sums log-likelihoods from `Landmark.calculate_likelihood()` for all matched landmarks. |
| `cleanup_old_landmarks(visible, landmark_index)` | Updates trust values and removes landmarks below threshold. |
| `publish_pointcloud_map(publisher, time)` | Publishes landmark odom positions and RGB colors as `sensor_msgs/PointCloud2`. |

---

#### tf_methods.py

Stateless coordinate frame transformation utilities. All transforms are computed
analytically using the calibrated rotation matrix `ROT_BK` and the fixed camera
offset `CAMERA_POS_IN_BASELINK`. No tf2_ros buffer lookups are performed.

| Function | From / To | Notes |
|----------|-----------|-------|
| `kinect_depth_to_odom(kinect_point, theta, pos_baselink)` | kinect to odom | Applies ROT_BK then heading rotation and translation. |
| `kinect_depth_to_baselink(kinect_point)` | kinect to base_link | Applies ROT_BK plus camera offset. |
| `odom_to_baselink(odom_point, theta, pos_baselink)` | odom to base_link | Inverse rotation by heading, subtract robot position. |
| `pixel_to_kinect(kp)` | pixel + depth to kinect 3D | Back-projects via pinhole model; output in meters. |
| `pixel_to_kinect_(u, v, z)` | pixel + depth to kinect 3D | Same as above, returns NDArray instead of Coordinate. |

---

#### visual_odom_node.py

Main ROS 2 node (`VisualOdom`). Orchestrates the particle filter, runs RANSAC
between consecutive frames, and broadcasts the pose of the best particle as TF.

**Initialization**

Parameters are loaded from the ROS 2 parameter server, backed by `param.yaml`.
A `cv2.ORB` detector and `cv2.BFMatcher` (NORM_HAMMING, crossCheck) are created.
N particles (`VisualRobotSample`) are instantiated on the first valid RGB frame.

**Callbacks**

| Callback | Description |
|----------|-------------|
| `listener_rgb_callback(msg)` | Main processing callback. Runs ORB detection, depth filtering, RANSAC, particle iteration, best-particle selection, and TF broadcast. |
| `listener_depth_callback(msg)` | Converts depth image to NumPy array and records timestamp for sync checks. |
| `listener_wheel_odom_callback(msg)` | Extracts pose from wheel odometry message and calls `predict_with_wheel_odom()` on all particles. |

**Key methods**

| Method | Description |
|--------|-------------|
| `ransac(matches, valid_kp, valid_kp_depth, valid_kp_last, valid_kp_depth_last)` | Frame-to-frame RANSAC: builds P (last frame 3D) and Q (current frame 3D) in base_link, iterates Kabsch hypotheses, refines on inliers. Returns `(translation, delta_theta, draw_keypoints, inlier_count)`. |
| `img_to_kp_des_filtered(msg)` | Detects ORB keypoints, filters by depth and pixel-neighborhood exclusion, computes descriptors. |
| `iteration(...)` | Runs `robot_iteration()` on all particles, normalizes log-weights, selects best particle. |
| `calculate_tf(pos, theta, timestamp, ...)` | Builds `TransformStamped` from current position and heading quaternion. |

---

#### visual_odom_launch.py

ROS 2 launch file. Reads `param.yaml` for the rosbag path and starts three
processes simultaneously:

| Process | Description |
|---------|-------------|
| `visual_odom_node` | Main odometry node with `param.yaml` parameters |
| `ros2 bag play` | Rosbag playback with `--clock` and topic filter for RGB, depth, and `/tf_static` |
| `rviz2` | Visualization |

---

### Configuration (param.yaml)

All parameters are declared under the `visual_odom/ros__parameters` namespace.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ransac.evaluation_tolerance` | Inlier distance threshold |
| `ransac.iterations` | Maximum RANSAC loop count |
| `ransac.sample_size` | Minimum point pairs per hypothesis |
| `ransac.min_inlier_ratio` | Minimum inlier fraction to accept RANSAC result |
| `ransac.min_matches_for_ransac` | Minimum matches required to start RANSAC |
| `ransac.max_rotation_angle_deg` | Maximum plausible rotation per frame in degrees |
| `rgb_depth_sync_tolerance_sec` | Max allowed RGB/depth timestamp delta |
| `pixel_tolerance` | Pixel-space exclusion radius for keypoint deduplication |
| `matches_for_new_landmarks` | Match count threshold below which new landmarks are added |
| `min_landmark_trust` | Trust value below which a landmark is pruned |
| `max_depth` | Maximum usable sensor depth |
| `n_robot_samples` | Number of particles in the particle filter |
| `rosbag_path` | Absolute path to the `.mcap` rosbag file |

---

## 3. Build and Run

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select visual_odom
source install/setup.bash
```

### Launch (recommended)

```bash
ros2 launch visual_odom visual_odom_launch.py
```

Starts the node, rosbag playback, and RViz2 simultaneously.
- The rosbag path is read automatically from `param.yaml`.
- bagfiles are saved in SLAM/bagfiles
- The rviz2 config in /rviz will automatically be loaded

### Node only (without launch file)

```bash
ros2 run visual_odom visual_odom_node --ros-args --params-file path/to/param.yaml
```

---

## 4. Documentation

The package uses [Doxygen](https://www.doxygen.nl) with Python docstring support.

### Generate HTML docs

```bash
doxygen Doxyfile
xdg-open docs/html/index.html
```

### Generate PDF from LaTeX output

```bash
cd docs/latex
make
xdg-open refman.pdf
```