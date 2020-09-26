# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:kalman_filter.py
# software: PyCharm

import numpy as np
import scipy.linalg


class KalmanFilter:

    def __init__(self):
        # motion and observation uncertainty are decide by std
        # these weights can control the uncertainty
        # TODO: i don't know the details of setting of these params
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        # transform matrix
        dim = 4
        self._transform_matrix = np.eye(2 * dim, 2 * dim)
        for i in range(4):
            self._transform_matrix[i, i + dim] = 1.0

        # update matrix
        self._update_matrix = np.eye(dim, 2 * dim)

    def initiate(self, measurement):
        """start tracking from unassociated measurement.

        Args:
            measurement: bounding boxes (x, y, a, h)
                         (x, y) center of bounding boxes
                         a, aspect ratio
                         h, height of boxes

        Returns:
            mean: initial mean
            cov: initial std

        """
        mean_position = measurement
        mean_velocity = np.zeros_like(mean_position)
        # (8,)
        mean = np.concatenate([mean_position, mean_velocity])
        # set std
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        # compute cov
        # except diagonal, elements are zero because the elements of mean are unassociated
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean, cov):
        """kalman filter prediction step.

        Args:
            mean: (mean_x, mean_y, mean_a, mean_h, mean_dx, mean_dy, mean_da, mean_dh) of previous step
            cov: the shape is (8, 8)

        Returns:

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        noise_cov = np.diag(np.square(std_pos))
        # transform mean and covariance
        mean = np.matmul(self._transform_matrix, mean)
        cov = np.linalg.multi_dot((self._transform_matrix, cov, self._transform_matrix.T)) + noise_cov

        return mean, cov

    def update(self, mean, cov, measurement):
        """start updating process with mean, cov and measurement
        kalman_gain = PK @ HK.T @ (HK @ PK @ HK.T + RK) ^ -1
        new_mean = mean + kalman_gain @ mean_diff
        new_cov = cov - kalman_gain @ HK @ cov
        PK - the covariance of current blur estimation
        HK - the transformation matrix that change estimation coordinate to measurement coordinate
             (x, y, a, h, dx, dy, da, dh) -> (x, y, a, h)
        RK - the covariance of detection results

        Args:
            mean:          [x, y, a, h, dx, dy, da, dh]
                           the mean of current blur estimation
            cov:           shape is (8, 8)
                           the covariance of current blur estimation
            measurement:   [x, y, a, h]
                           the results of detector

        Returns:
            new_mean:      (8,)
            new_cov:       (8, 8)

        """
        # std of measurement
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        measure_cov = np.diag(np.square(std))  # (4, 4)
        mean_middle = np.dot(self._update_matrix, mean)  # (4,)
        cov_middle = np.linalg.multi_dot((self._update_matrix, cov, self._update_matrix.T)) + measure_cov  # (4, 4)
        # mean_diff = zk - Hk @ xk
        mean_diff = measurement - mean_middle  # (4,)
        # kalman gain
        # Ax = B, solve x
        chol_factor, lower = scipy.linalg.cho_factor(cov_middle, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(cov, self._update_matrix.T).T,
                                             check_finite=False).T  # (8, 4)
        new_mean = mean + np.dot(kalman_gain, mean_diff)  # (8,)
        new_conv = cov - np.linalg.multi_dot((kalman_gain, self._update_matrix, cov))  # (8, 8)

        return new_mean, new_conv

    def maha_distance(self,
                      mean,
                      cov,
                      measures,
                      only_position=True):
        """compute maha-distance between optimal estimation by kalman filter and measures.

        maha_distance = (dj - yi).T @ Si ^ -1 @ (dj - yi)
        yi, Si - track distribution into measurement space
        dj - the j-th bounding box detection

        why i use maha distance?
        Because Euclid distance have a disadvantage that it does not consider correlation between
        params.
        Maha-distance can discriminate outlier better than Euclid distance.
        |
        |    *-1           1 is outlier but Euclid distance of 1 is equal to 2
        |           *-2
        |        *****
        |       *****
        |      ****
        |     ****
        -----------------------------


        Args:
            mean:           (x, y, a, h, dx, dy, da, dh)
                            corrected state distribution
            cov:            shape is (8, 8)
                            corrected state distribution
            measures:       detection results
            only_position:  if only position, only compute center of bounding boxes

        Returns:
            maha_distances: a list of maha_distance

        """
        # middle mean
        mean_middle = np.dot(self._update_matrix, mean)
        # middle cov
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        innovation_cov = np.diag(np.square(std))
        covariance = np.linalg.multi_dot((self._update_matrix, cov, self._update_matrix.T))
        cov_middle = covariance + innovation_cov  # (4, 4)
        # A @ A^-1 = identity_matrix
        identity_matrix = np.eye(4, 4)

        if only_position:
            mean_middle = mean_middle[..., :2]
            cov_middle = cov_middle[:2, :2]
            measures = np.array(measures)[..., :2]
            identity_matrix = np.eye(2, 2)

        # get maha distance
        cholesky_factor, lower = scipy.linalg.cho_factor(cov_middle)
        cov_middle_inverse = scipy.linalg.cho_solve((cholesky_factor, lower),
                                                    identity_matrix,
                                                    check_finite=False)

        maha_distances = []
        num_measure = np.shape(measures)[0]
        for i in range(num_measure):
            measure = measures[i]
            diff = measure - mean_middle
            maha_distance = np.linalg.multi_dot((diff.T, cov_middle_inverse, diff))
            maha_distances.append(maha_distance)

        return maha_distances


if __name__ == '__main__':
    # test my kalman filter

    measurement_ = np.array([100.0, 50.0, 2.0, 30.0])

    filter_ = KalmanFilter()

    # initialize filter
    mean_init, cov_init = filter_.initiate(measurement=measurement_)
    # print(mean_init, cov_init.dtype)

    # start predicting
    mean_pred, cov_pred = filter_.predict(mean_init, cov_init)
    # print(mean_pred, cov_pred.shape)

    # start updating
    mean_update, cov_update = filter_.update(mean_pred, cov_pred, measurement_)
    print(mean_update.shape, cov_update.shape)

    # maha_distance
    measurement_next = np.expand_dims(measurement_, axis=0)
    distance = filter_.maha_distance(mean_update, cov_update, measurement_next)
    print(distance)
