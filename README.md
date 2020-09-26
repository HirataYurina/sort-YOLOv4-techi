# Simple Online Realtime Tracking

> A simple algorithm for multi-objects tracking.

### 1. Detector

The detector can be any model that can detect objects you want to track , such as person, car or animal.
Outputs of detector should be: [x, y, a, h].
x, y - the center of bounding box;
a - aspect ratio(w / h);
> if you set ratio=h/w, the maha distance will be larger. And this will cause the maha distance become bigger.
>
> So, in sort algorithm, author set aspect to be constant.
>
> And, in my experiment i don't consider a and h in computing gating distance.

h - height of bounding box.

### 2. Tracking Method

#### 2.1 Estimation Model

1. The first frame should be initialized:

   measurement: [x, y, a, h] --> mean: [x, y, a, h, dx, dy, da, dy] & covariance (8, 8).

2. The initial of mean is [x, y, a, h, 0, 0, 0, 0].

3. Propagate current frame state into the next frame by using a linear constant velocity model.

   **If a target is associated to a detection(measurement), then update the target with kalman filter.**

   **If no detection is associated to the target, just predict the target with linear velocity model without correcting.**

#### 2.2 Data Association

1. Predicet next frame state from current frame state through linear velocity model.

2. The data distribution will be changed through transforming, and we call this prediction.
   $$
   mean'=Fk*mean
   $$

   $$
   cov'=Fk*cov*Fk^T
   $$

3. Compute maha distance between **prediction of this frame** and detection of this frame.

4. Assign matched detection to target that we track.

5. Update prediction if it has been matched with kalman gain.

6. **Use iou matching to match unconfirmed trackers of age = 1 to remain unmatched detections.**

#### 2.3 Create and Delete Trackers

1. **Delete trackers** of age > 30 or tentative trackers of age > 3.

2. **Create trackers** with unmatched detections at this frame.

### 3. Deep Sort

There is a problem in SORT:

if motion uncertainty is slow, maha distance is a comfortable metric.

But, if motion uncertainty is big, maha distance is not stable.

So, DeepSort add a performance descriptor to promote tracking stability.

### 4. ToDO

- [ ] DeepSort

  

