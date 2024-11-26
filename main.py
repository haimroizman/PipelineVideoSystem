import multiprocessing as mp
import logging
import os
from streamer import Streamer
from detector import Detector
from presenter import Presenter

# Setup main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Main')


def get_video_path():
    """
    Returns a valid video path. You can use:
    1. Webcam: 0 for default camera
    2. Video file: 'path/to/video.mp4'
    3. IP camera: 'rtsp://username:password@ip_address:port/path'
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, 'sample_videos', 'People.mp4')

    logger.info(f"Looking for video at: {video_path}")
    return video_path


def main():
    try:
        logger.info("Starting video analytics pipeline")

        # Create communication queues with size limit to prevent memory issues
        streamer_to_detector = mp.Queue(maxsize=10)
        detector_to_presenter = mp.Queue(maxsize=10)

        # Get video path
        video_path = get_video_path()
        print(f'video_path: {video_path}')
        if not isinstance(video_path, int) and not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create processes
        streamer = Streamer(video_path, streamer_to_detector)
        detector = Detector(streamer_to_detector, detector_to_presenter)
        presenter = Presenter(detector_to_presenter)

        logger.info("Starting processes")
        # Start processes in order
        streamer.start()
        detector.start()
        presenter.start()

        # Wait for all processes to complete
        streamer.join()
        detector.join()
        presenter.join()

        logger.info("All processes completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

    finally:
        # Clean up any remaining processes
        for process in [streamer, detector, presenter]:
            if process.is_alive():
                process.terminate()
                logger.info(f"Terminated {process}")


if __name__ == "__main__":
    # Required for Windows support
    mp.freeze_support()
    main()
