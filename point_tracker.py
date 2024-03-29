import cv2 as cv 
import numpy as np
import json
import argparse
import os

# Global variables to store points and their IDs
selected_points = []
point_id_counter = 0
point_id_map = {}
points_data_all_frames = {}

def select_points(event, x, y, flags, param):
    global selected_points, point_id_counter
    frame = param
    if event == cv.EVENT_LBUTTONDOWN:
        point_id_counter += 1
        selected_points.append((point_id_counter, x, y))
        point_id_map[point_id_counter] = (x, y)
        for point in selected_points:
            cv.circle(frame, (point[1], point[2]), 5, (128, 255, 0), -1)
            cv.putText(frame, str(point[0]), (point[1] + 10, point[2] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('Select Points', frame)

def lucas_kanade(video_path, output_video_path, json_file_path, create_video=True): 
    """
    Track points in a video using the Lucas-Kanade method.

    Args:
    - video_path (str): Path to the input video file.
    - output_video_path (str): Path to save the output tracked video file.
    - json_file_path (str): Path to save the tracked points data in JSON format.
    - create_video (bool, optional): Flag to enable or disable the creation of the output video. Default is True.

    The function reads the input video file from 'video_path' and tracks points using the Lucas-Kanade method.
    It allows the user to select points on the first frame interactively.
    The tracked points are visualized on the video frames.
    The tracked points data for each frame is saved to a JSON file specified by 'json_file_path'.
    If 'create_video' is True, the function also saves the video with tracked points to 'output_video_path'.
    """
    # Read video 
    video_cap_obj = cv.VideoCapture(video_path)
    frame_width = int(video_cap_obj.get(3))
    frame_height = int(video_cap_obj.get(4))
    fps = int(video_cap_obj.get(cv.CAP_PROP_FPS))

    lucas_kanade_params = dict(winSize=(30, 30),
                             maxLevel=3,
                             criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find features to track 
    _, frame_first = video_cap_obj.read()
    frame_gray_prev = cv.cvtColor(frame_first, cv.COLOR_BGR2GRAY)

    # Allow user to select points on the first frame
    global selected_points, point_id_map
    selected_points = []
    point_id_map = {}
    cv.namedWindow('Select Points', cv.WINDOW_NORMAL)  # Set window to be resizable
    cv.setMouseCallback('Select Points', select_points, param=frame_first)
    while True:
        cv.imshow('Select Points', frame_first)
        cv.setWindowProperty('Select Points', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)  # Set window to full screen
        key = cv.waitKey(10)
        if key == ord('q'):
            break
    cv.destroyAllWindows()

    corners_prev = np.array([(point[1], point[2]) for point in selected_points], dtype=np.float32).reshape(-1, 1, 2)

    # Define the codec and create VideoWriter object if create_video is True
    out = None
    if create_video:
        out = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_points_dict = {}
    # Loop through each video frame 
    frame_index = 0
    while True: 
        ret, frame = video_cap_obj.read()
        if not ret:
            break
        frame_gray_cur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners_cur, found_status, _ = cv.calcOpticalFlowPyrLK(frame_gray_prev, frame_gray_cur, corners_prev, None, **lucas_kanade_params)

        for i, cur_corner in enumerate(corners_cur):
            if found_status[i] == 0:
                continue
            x_cur, y_cur = cur_corner.ravel()
            cv.circle(frame, (int(x_cur), int(y_cur)), 5, (128, 255, 0), -1)
            cv.putText(frame, str(selected_points[i][0]), (int(x_cur) + 10, int(y_cur) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            if frame_index not in frame_points_dict:
                frame_points_dict[frame_index] = []
            frame_points_dict[frame_index].append({selected_points[i][0]: {'x': int(x_cur), 'y': int(y_cur)}})

        if create_video:
            out.write(frame)  # Write frame with tracked points to video

        cv.imshow('Video', frame)
        cv.setWindowProperty('Video', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)  # Set window to full screen
        cv.waitKey(15)

        frame_index += 1
        frame_gray_prev = frame_gray_cur.copy()
        corners_prev = corners_cur.reshape(-1, 1, 2)
    
    # Release VideoWriter object if it was created
    if out:
        out.release()

    # Release VideoCapture object
    video_cap_obj.release()

    # Close all OpenCV windows
    cv.destroyAllWindows()

    # Save points data to a json file
    with open(json_file_path, 'w') as f:
        json.dump(frame_points_dict, f, indent=4)

def get_output_video_path(video_path):
    # Get base file name without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    return f"{base_name}_tracked.mp4"

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Track points using Lucas-Kanade method')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('-o', '--output_video', type=str, default='', help='Path to the output tracked video file')
    parser.add_argument('-j', '--json_file', type=str, default='', help='Path to the JSON file to save the tracked points data')
    parser.add_argument('-n', '--no_video', action='store_true', help='Flag to disable output video creation')
    args = parser.parse_args()

    output_video_path = args.output_video if args.output_video else get_output_video_path(args.video_path)
    json_file_path = args.json_file if args.json_file else output_video_path.split('.')[0] + '_points.json'

    lucas_kanade(args.video_path, output_video_path, json_file_path, not args.no_video)
