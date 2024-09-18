import cv2
import torch
import mediapipe as mp
import numpy as np
from scipy.interpolate import RectBivariateSpline

# function to detect gpu device
def detect_gpu():
    device = ""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Utilizing Device: {device}")
    return device


# function to return distance of objects
def depth_to_distance(depth_value, depth_scale):
    return -1.0/(depth_value*depth_scale)


# Setting up pipeline
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Loading Models
'''
    Small variant: 'MiDaS_small'
    Hybrid variant: 'DPT_Hybrid'
    Large variant:  'DPT_Large'
'''
model_type = 'MiDaS_small'
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = detect_gpu()
midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = transforms.dpt_transform
else:
    transform = transforms.small_transform


cap = cv2.VideoCapture('test_videos/Walking.mp4')
while cap.isOpened():
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Walking', img)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()



