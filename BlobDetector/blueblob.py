#!/usr/bin/python

# Standard imports
import cv2
import numpy as np


class BlueDotDetector:
    def __init__(self):
        """ Create blob detector """

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # https://github.com/opencv/opencv/blob/master/modules/features2d/src/blobdetector.cpp#L83

        # do our own color filtering https://stackoverflow.com/questions/31460267/python-opencv-color-tracking/31465462
        params.filterByColor = False
        color = 110
        sens = 25
        self.lower_hsv = np.array([color-sens, 100, 50])
        self.upper_hsv = np.array([color+sens, 255, 255])

        # # Change thresholds
        # params.minThreshold = 10
        # params.maxThreshold = 200

        # # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 25
        params.maxArea = 90000

        # # Filter by Circularity
        # params.filterByCircularity = False
        # params.minCircularity = 0.1

        # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87

        # # Filter by Inertia
        # params.filterByInertia = True
        params.minInertiaRatio = 0.001

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_frame = cv2.inRange(hsv_frame, self.lower_hsv, self.upper_hsv)
        cv2.imshow("Mask Frame", mask_frame)
        keypoints = self.detector.detect(mask_frame)
        print(keypoints)
        return keypoints

# --------------------------------------------------------------------------


if __name__ == '__main__':
    print('Press "q" to quit')
    capture = cv2.VideoCapture(0)

    bd = BlueDotDetector()

    if capture.isOpened():  # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False

    while frame_captured:
        keypoints = bd.detect(frame)
        marked_frame = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('Test Frame', frame)
        cv2.imshow("Marked Frame", marked_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
