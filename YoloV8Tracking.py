import shutil

from ultralytics import YOLO
import cv2
import os
from PhotoToVideo import images_to_video
model = YOLO('runs/detect/train10/weights/best.pt')

folder_path = "yolo_temp_frames"
try:
    # delete folder if it exists
    ok = False
    counter = 0
    while not ok:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            counter += 1
            idx = folder_path.find('(')
            if idx != -1:
                folder_path = folder_path[:idx] + f"({counter})"
            else:
                folder_path += f"({counter})"
        else:
            ok = True
    # create empty folder for temp frames
    os.makedirs(folder_path)
    #print(f"Folder created at: {folder_path}")
except OSError as error:
    print(f"Error creating temp folder: {error}")


#video_path = 'frisbee_videos/stable_throws/stable_throws_clip_4.mp4'
video_path = "frisbee_videos/red_tape_throws/gopro_frisbee_clip_14.mp4"
cap = cv2.VideoCapture(video_path)
ret = True
frame_counter = 0
detected_data = []
while ret:
    ret, frame = cap.read()
    try:
        results = model.track(frame, persist=True)  # remembers the objects- helpful maybe?
    except:
        break

    frame_ = results[0].plot()
    frame_filename = os.path.join(folder_path, f"frame_{frame_counter}.png")
    cv2.imwrite(frame_filename, frame_)
    cv2.imshow('frame', frame_)

    # Print details of each detection
    all_bb_info = []
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls)
        confidence = float(box.conf)

        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        bounding_box_info = [x1, x2, y1, y2, center_x, center_y]
        all_bb_info.append(bounding_box_info)

    frame_info = (frame_counter, all_bb_info)
    detected_data.append(frame_info)

    frame_counter += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Clean Up
output_filename = "yolo_tracking_"+video_path[video_path.rfind('/')+1:]
images_to_video(folder_path, output_filename, 20, ".png")

try:
    shutil.rmtree(folder_path)
except Exception as error:
    print(f"Error deleting folder: {error}")