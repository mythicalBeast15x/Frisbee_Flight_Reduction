import os
import numpy as np
import shutil
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.io import write_jpeg
from torchvision.utils import flow_to_image
from PhotoToVideo import images_to_video
from HelperFuncions import print_progress_bar
import cv2
import cv2
import torch
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PhotoToVideo import images_to_video
import shutil
from HelperFuncions import print_progress_bar

import time
def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()


def preprocess(im1_batch, im2_batch):
    im1_batch = F.resize(im1_batch, size=[520, 920], antialias=False)
    im2_batch = F.resize(im2_batch, size=[520, 920], antialias=False)
    return transforms(im1_batch, im2_batch)


DELETE_RAFT_FRAMES = False

# create temporary folder for frames
folder_path = "raft_temp_frames"
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

plt.rcParams["savefig.bbox"] = "tight"

video_path = "frisbee_videos/stable_throws/stable_throws_clip_11.mp4"
s_time = time.time()
# get frames from video
frames, _, _ = read_video(str(video_path), output_format="TCHW")
#frames = frames[58:112]
"""frames = []
video = cv2.VideoCapture(video_path)
total_frames = 0
while True:
    # Read a frame
    ret, frame = video.read()
    if not ret:
        break
    frames.append(frame)
    total_frames += 1
video.release()"""

e_time = time.time()

print(e_time-s_time, "seconds")

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print(device)


model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

print("Predictions:")
total_frames = len(list(zip(frames, frames[1:])))
start_time = time.time()
for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
    img1, img2 = preprocess(img1.unsqueeze(0), img2.unsqueeze(0))
    list_of_flows = model(img1.to(device), img2.to(device))
    predicted_flow = list_of_flows[-1][0]
    flow_img = flow_to_image(predicted_flow).to("cpu")
    #output_folder = "Notebooks/predicted_flows/"
    #write_jpeg(flow_img, folder_path+f"predicted_flow_{i}.jpg")
    write_jpeg(flow_img, os.path.join(folder_path,f"predicted_flow_{i}.jpg"))
    print_progress_bar(i+1, total_frames, prefix='Progress:', suffix='Complete', start_time=start_time)
#print("\n")
# POST PREDICTION

# Recreate Video
raft_video_name = video_path[video_path.rfind('/')+1:]
raft_video_name = raft_video_name[:raft_video_name.rfind('.mp4')] + "_RAFT-DEMO-test.mp4"
images_to_video(folder_path, raft_video_name)

# Clean up
if DELETE_RAFT_FRAMES:
    try:
        shutil.rmtree(folder_path)
    except Exception as error:
        print(f"Error deleting folder: {error}")




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

filename = "midas_"+raft_video_name
cap = cv2.VideoCapture(video_path)
total_frames = 0
if cap.isOpened():
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
else:
    print("Video Path does not exist")  # TODO add return statement here in function


# create temporary folder for frames
folder_path = "midas_temp_frames"
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
output_filename = filename[:-4]+"midas_1"+filename[-4:]
images_to_video(folder_path, output_filename, 15, ".png")

# Clean up
try:
    shutil.rmtree(folder_path)
except Exception as error:
    print(f"Error deleting folder: {error}")


#MIDAS SECTION


"""# looks for gpu then returns device or cpu if none is detected
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

#filename = "gopro_frisbee_clip_8.mp4"
filename = raft_video_name
cap = cv2.VideoCapture(filename)
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
"""
# Clean up
"""try:
    shutil.rmtree(folder_path)
except Exception as error:
    print(f"Error deleting folder: {error}")"""
