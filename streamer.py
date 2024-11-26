import cv2
import multiprocessing as mp
import time
import logging
from logging.handlers import RotatingFileHandler


class Streamer(mp.Process):
    def __init__(self, video_path, output_queue):
        super().__init__()
        self.video_path = video_path
        self.output_queue = output_queue
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging for this process"""
        logger = logging.getLogger(f'Streamer-{self.name}')
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('streamer.log', maxBytes=1024 * 1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def run(self):
        '''
        Main process funtion that runs in a separate procees. Continuously reads frames from the video file and
        sends it to the detector process.
        '''
        try:
            self.logger.info(f"Starting streamer process with video: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video stream reached")
                    self.output_queue.put(None)
                    break

                # Send frame to detector
                self.output_queue.put(frame)
                # Small delay to prevent overwhelming the queue
                time.sleep(0.01)

        except Exception as e:
            self.logger.error(f"Error in streamer process: {str(e)}")
            self.output_queue.put(None)
        finally:
            cap.release()
            self.logger.info("Streamer process ended")

    # def __str__(self):
    #     return f"Streamer(video_path={self.video_path})"
