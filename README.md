# Simple Online Realtime Tracking

> A simple algorithm for multi-objects tracking.

### Detector

The detector can be any model that can detect objects you want to track , such as person, car or animal.
Outputs of detector should be: [x, y, a, h].
x, y - the center of bounding box;
a - aspect ratio(h / w);
h - height of bounding box.

### Tracking Method

#### Estimation Model

1. The first frame should be initialized:

   measurement: [x, y, a, h] --> mean: [x, y, a, h, dx, dy, da, dy] & covariance (8, 8).

2. The initial of mean is [x, y, a, h, 0, 0, 0, 0].

3. Propagate current frame state into the next frame by using a linear constant velocity model.

   **If a target is associated to a detection(measurement), then update the target with kalman filter.**

   **If no detection is associated to the target, just predict the target with linear velocity model without correcting.**

#### Data Association

1. Predicet next frame state from current frame state through linear velocity model.

2. The data distribution will be changed through transforming, and we call this prediction.
   $$
   mean'=Fk*mean
   $$

   $$
   cov'=Fk*cov*Fk^T
   $$

3. Compute maha distance between prediction of this frame and detection of this frame.

4. Assign matched detection to target that we track.

5. Update prediction if it has been matched with kalman gain.

6. **Use iou matching to match unconfirmed trackers of age = 1 to remain unmatched detections.**

#### Create and Delete Trackers

1. **Delete trackers** of age > 30 or tentative trackers of age > 3.
2. **Create trackers** with unmatched detections at this frame.

  

