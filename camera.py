import cv2

class VideoCameraModel(object):
    def __init__(self,):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def set_frame_rate(self, fps):
        return self.video.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self):
        # Get picture from video stream
        success, image = self.video.read()
        return image

