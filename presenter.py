import cv2
import multiprocessing as mp
from datetime import datetime
import logging
import numpy as np
from logging.handlers import RotatingFileHandler


class Presenter(mp.Process):
    def __init__(self, input_queue):
        super().__init__()
        self.input_queue = input_queue
        self.logger = self._setup_logger()
        # Pre-compute Gaussian kernel for better performance
        self.kernel = self._create_gaussian_kernel(25, sigma=3)

    def _setup_logger(self):
        logger = logging.getLogger(f'Presenter-{self.name}')
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('presenter.log', maxBytes=1024 * 1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _create_gaussian_kernel(self, size, sigma=3):
        """
        Create a 2D Gaussian kernel using NumPy.

        Parameters:
        - size: Kernel size (must be odd)
        - sigma: Standard deviation of Gaussian distribution

        Returns:
        - 2D numpy array representing the Gaussian kernel

        """
        # Ensure size is odd
        size = size + 1 if size % 2 == 0 else size

        # Create 1D coordinates
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        # Create 2D coordinates using meshgrid
        xx, yy = np.meshgrid(ax, ax)

        # Calculate Gaussian values
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

        # Normalize kernel
        return kernel / np.sum(kernel)

    def apply_gaussian_blur(self, frame, regions, kernel_size=25):

        try:
            # Work with frame as numpy array directly, using Numpy's optimized operations...
            frame_copy = np.array(frame, dtype=np.float32)

            for (x, y, w, h) in regions:
                # Extract region of interest (ROI)
                roi = frame_copy[y:y + h, x:x + w]

                if roi.size == 0:
                    continue

                # Apply blur using NumPy's optimized convolution
                # As I understood cv2.filter2D is more efficient than manual convolution
                blurred_roi = cv2.filter2D(roi, -1, self.kernel)

                # Update the ROI directly
                frame_copy[y:y + h, x:x + w] = blurred_roi

            # Convert back to uint8 for display
            return np.uint8(frame_copy)

        except Exception as e:
            self.logger.error(f"Error in NumPy Gaussian blur: {str(e)}")
            return frame

    def draw_detections(self, frame, motion_regions):
        """
        Draw timestamp and apply optimized Gaussian blur to motion regions.
        """
        try:
            # Apply Gaussian blur to motion regions
            blurred_frame = self.apply_gaussian_blur(frame, motion_regions)

            # Draw timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(blurred_frame, timestamp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return blurred_frame

        except Exception as e:
            self.logger.error(f"Error drawing detections: {str(e)}")
            return frame

    def run(self):
        self.logger.info("Starting presenter process")
        try:
            while True:
                result = self.input_queue.get()
                if result is None:
                    self.logger.info("Received end of stream signal")
                    break

                frame, motion_regions = result
                output_frame = self.draw_detections(frame, motion_regions)
                cv2.imshow("Video Analytics", output_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("User requested quit")
                    break

        except Exception as e:
            self.logger.error(f"Error in presenter process: {str(e)}")

        finally:
            cv2.destroyAllWindows()
            self.logger.info("Presenter process terminated")