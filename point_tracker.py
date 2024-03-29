import cv2 
import numpy as np
import json
import argparse


class LucasKanadeTracker:
    def __init__(self, first_frame, selected_points,
                 win_size = 30, max_level = 10, criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 5, 0.05)):
        """
        Initialize the Lucas-Kanade tracker with the first frame and selected points.
        
        Args:
        - first_frame (numpy.ndarray): The first frame of the video.
        - selected_points (list): List of selected points to track. Each point is a tuple (x, y).
        - win_size (int, optional): Size of the window for the Lucas-Kanade method. Default is 30.
        - max_level (int, optional): Maximum pyramid level number. Default is 3.
        - criteria (tuple, optional): Criteria for the termination of the iterative process. Default is (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03).
        """
        self.frame_gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.points_prev = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)
        self.win_size = win_size
        self.max_level = max_level
        self.criteria = criteria

    def track(self, frame_cur):
        """
        Track points in the current frame using the Lucas-Kanade method.

        Args:
        - frame_cur (numpy.ndarray): The current frame of the video.

        Returns:
        - points_cur_converted (list): List of tracked points in the current frame. Each point is a tuple (x, y).
        - status (numpy.ndarray): Status array indicating whether the flow for the corresponding points has been found. 1 indicates found, 0 indicates not found.
        """
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        points_cur, status, _ = cv2.calcOpticalFlowPyrLK(self.frame_gray_prev, frame_gray_cur, self.points_prev, None, winSize=(self.win_size, self.win_size), maxLevel=self.max_level, criteria=self.criteria)
        self.frame_gray_prev = frame_gray_cur.copy()
        self.points_prev = points_cur.reshape(-1, 1, 2)

        points_cur_converted = []
        for i, point_cur in enumerate(points_cur):
            x_cur, y_cur = point_cur.ravel()
            point_cur = (int(x_cur), int(y_cur))
            points_cur_converted.append(point_cur)

        return points_cur_converted, status


def select_points_on_frame(frame):
    selected_points = []
    def _select_points(event, x, y, flags, param):
        nonlocal selected_points
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            for id, point in enumerate(selected_points):
                cv2.circle(frame, point, 5, (128, 255, 0), -1)
                cv2.putText(frame, str(id), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Select Points', frame)
    
    cv2.namedWindow('Select Points', cv2.WINDOW_NORMAL)  # Set window to be resizable
    cv2.setMouseCallback('Select Points', _select_points, param=frame)
    while True:
        cv2.imshow('Select Points', frame)
        cv2.setWindowProperty('Select Points', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    return selected_points


def get_first_video_frame(video_path):
    video_cap_obj = cv2.VideoCapture(video_path)
    ret, first_frame = video_cap_obj.read()
    return first_frame


def draw_point_on_frame(frame, point, id):
    cv2.circle(frame, point, 5, (128, 255, 0), -1)
    cv2.putText(frame, str(id), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def track_points(video_path, save=False):
    """
    Track points in the video using the Lucas-Kanade method. The user can select points to track on the first frame of the video.
    
    Args:
    - video_path (str): Path to the input video file.
    - save (bool, optional): Flag to save the tracked points data, and the output video with tracked points. Default is False.
    """
    first_frame = get_first_video_frame(video_path)
    selected_points = select_points_on_frame(first_frame.copy())
    
    lucas_kanade_tracker = LucasKanadeTracker(first_frame, selected_points)

    # Load video
    video_cap_obj = cv2.VideoCapture(video_path)
    frame_width = int(video_cap_obj.get(3))
    frame_height = int(video_cap_obj.get(4))
    fps = int(video_cap_obj.get(cv2.CAP_PROP_FPS))

    # Initialization
    ret, first_frame = video_cap_obj.read()
    for id, point in enumerate(selected_points):
        draw_point_on_frame(first_frame, point, id)
    cv2.imshow('Video', first_frame)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Wait for a bit to show the first frame
    cv2.waitKey(500)

    wait_time = int(1/fps * 1000)

    if save:
        out = None
        json_data = {}
        output_video_path = video_path.split('.')[0] + '_tracked.mp4'
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        out.write(first_frame)
        json_data[0] = {id: {'x': x, 'y': y} for id, (x, y) in enumerate(selected_points)}

    frame_index = 1

    # Continue tracking points in the video
    while True:
        ret, frame = video_cap_obj.read()
        if not ret:
            break

        points_cur, status = lucas_kanade_tracker.track(frame)

        for i, point_cur in enumerate(points_cur):
            if status[i] == 0:
                continue
            draw_point_on_frame(frame, point_cur, i)
        
        cv2.imshow('Video', frame)
        cv2.waitKey(wait_time)

        if save:
            out.write(frame)
            json_data[frame_index] = {id: {'x': x, 'y': y} for id, (x, y) in enumerate(points_cur) if status[id] == 1}

        frame_index += 1
    
    video_cap_obj.release()
    cv2.destroyAllWindows()
    
    if save:
        out.release()
        json_file_path = video_path.split('.')[0] + '_points.json'
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)



if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Track points using Lucas-Kanade method')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('-s', '--save', action='store_true', help='Flag to save the tracked points data, and the output video with tracked points')

    args = parser.parse_args()
    track_points(args.video_path, args.save)