# Point Tracker

This repository contains a Python script for point tracking in videos using the Lucas-Kanade method. Users can select points interactively on the first frame of a video, and the script tracks these points across subsequent frames, visualizing them on the video. It also provides options to save the tracked points data to a JSON file and create an output video with tracked points.

## Demo

https://github.com/ippen/point-tracker/assets/103518421/bc487559-819c-4f8d-8121-33ea7627517a

## Features

- Interactive point selection: Users can select points on the first frame of the video by clicking with the mouse.
- Visualization: Tracked points are visualized on each frame of the video.
- Output options: The script can create an output video with tracked points and save the tracked points data to a JSON file.
- Command-line interface: Easy-to-use command-line interface with options to specify input and output files.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Usage

1. Clone the repository
    ```
    git clone https://github.com/ippen/point-tracker.git
    ```

2. Install the required dependencies using pip:

    ```
    pip install opencv-python numpy
    ```

3. Run the script with the following command:

    ```
    python point_tracker.py <video_path> [options]
    ```

    Replace `<video_path>` with the path to your input video file.

## Command-line Options

- `video_path`: Path to the input video file.
- `-s, --save`: Optional. If set, saves tracked points to a JSON file (`<video_name>_points.json`) and the output video with tracked points to an MP4 file (`<video_name>_tracked.mp4`).

## Example

```
python point_tracker.py input_video.mp4 --save
```

This command will track points in `input_video.mp4` and save the tracked points data and the output video with tracked points if the `--save` flag is set.