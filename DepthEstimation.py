import cv2
import torch
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PhotoToVideo import images_to_video
import shutil
from HelperFuncions import print_progress_bar


# looks for gpu then returns device or cpu if none is detected
def get_device():
    if torch.cuda.is_available():
        gpu_device = "cuda"
    elif torch.backends.mps.is_available():
        gpu_device = "mps"
    else:
        gpu_device = "cpu"
    print(f"Utilizing device: {gpu_device}")


def predict(frame, frames_depth_data, model):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    frames_depth_data.append(output)

    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    output = (output * 255).astype(np.uint8)

    return cv2.applyColorMap(output, cv2.COLORMAP_MAGMA)

# really cool progress bar



model_type = "DPT_Large"
#model_type = "DPT_Hybrid"
#model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = get_device()
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

filename = "gopro_frisbee_clip_8.mp4"
cap = cv2.VideoCapture("frisbee_videos/red_tape_throws/gopro_frisbee_clip_8.mp4")
total_frames = 0
if cap.isOpened():
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
else:
    print("Video Path does not exist")  # TODO add return statement here in function


# create temporary folder for frames
folder_path = "temp_frames"
try:
    # delete folder if it exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and its contents have been deleted.")
        except Exception as error:
            print(f"Error deleting folder: {error}")
    # create empty folder for temp frames
    os.makedirs(folder_path)
    #print(f"Folder created at: {folder_path}")
except OSError as error:
    print(f"Error creating temp folder: {error}")

frames_data = []
counter = 0
# create depth maps from video
while True:
    ret, frame = cap.read()

    if ret:

        depth_map = predict(frame, frames_data, midas)
        #cv2.imshow("Depth Map", depth_map)

        frame_filename = os.path.join(folder_path, filename[:-4]+f"frame_{counter}.png")
        cv2.imwrite(frame_filename, depth_map)
        print_progress_bar(counter+1, total_frames, prefix='Progress:', suffix='Complete')
        counter += 1
        # Press 'q' to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

# Reconstruct video from frames
output_filename = filename[:-4]+"_depth_map"+filename[-4:]
images_to_video(folder_path, output_filename, 30, ".png")

# Clean up
try:
    shutil.rmtree(folder_path)
except Exception as error:
    print(f"Error deleting folder: {error}")
