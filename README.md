# Human Detection Application with HOG and OpenCV

This project implements a human detection application using the Histogram of Oriented Gradients (HOG) feature descriptor and OpenCV. The application captures video from a webcam, detects people in real-time, and displays a count of the detected individuals.

## Features

- Real-time human detection using HOG.
- Displays bounding boxes around detected persons.
- Counts the number of detected individuals and displays the count on the video feed.
- Saves the output video with detections to `output.avi`.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. install libraries.

```bash
pip install numpy opencv-python
```

2. Run
```bash
python3 main.py 
```