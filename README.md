# Video Analytics Pipeline

A multi-process video analytics system that detects and blurs motion in video streams.

## Features

- Real-time video stream processing
- Motion detection
- Gaussian blur on detected regions
- Multi-process architecture using Python's multiprocessing
- Timestamp overlay on processed frames

## Project Structure

```
PipelineSystem/
├── main.py           # Main entry point
├── streamer.py       # Video streaming component
├── detector.py       # Motion detection component
├── presenter.py      # Visualization and blur effects
├── requirements.txt  # Project dependencies
└── sample_videos/    # Directory for test videos
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/video-analytics-pipeline.git
cd video-analytics-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your video file in the `sample_videos` directory
2. Run the main script:
```bash
python main.py
```

3. Press 'q' to quit the application

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- imutils

## License

[Choose an appropriate license for your project]