import cv2
import imutils
import multiprocessing as mp
import logging
from logging.handlers import RotatingFileHandler


class Detector(mp.Process):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f'Detector-{self.name}')
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('detector.log', maxBytes=1024 * 1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def detect_motion(self, frame, prev_frame):
        try:
            # Convert frames to grayscale -> Grayscale is sufficient for motion detection and cv2.COLOR_BGR2GRAY converts from BGR color space to single-channel grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is None:
                return [], gray_frame

            # Compute difference between frames
            diff = cv2.absdiff(gray_frame, prev_frame)
            # cv2.threshold converts grayscale to binary image parameters: (image, threshold_value, max_value, threshold_type)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            # Expand white regions to connect nearby motion regions
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours -> Contours are boundaries of white regions
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # Filter contours by area
            motion_regions = []
            for c in cnts:
                if cv2.contourArea(c) > 500:  # Minimum area threshold
                    (x, y, w, h) = cv2.boundingRect(c)
                    motion_regions.append((x, y, w, h))

            return motion_regions, gray_frame

        except Exception as e:
            self.logger.error(f"Error in motion detection: {str(e)}")
            return [], None

    def run(self):
        self.logger.info("Starting detector process")
        prev_frame = None
        try:
            while True:
                frame = self.input_queue.get()
                if frame is None:
                    self.logger.info("Received end of stream signal")
                    self.output_queue.put(None)
                    break

                motion_regions, gray_frame = self.detect_motion(frame, prev_frame)
                prev_frame = gray_frame
                self.output_queue.put((frame, motion_regions))

        except Exception as e:
            self.logger.error(f"Error in detector process: {str(e)}")
            self.output_queue.put(None)

        finally:
            self.logger.info("Detector process terminated")
