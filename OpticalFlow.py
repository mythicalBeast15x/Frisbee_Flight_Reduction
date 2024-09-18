import os
import sys
sys.path.append('RAFT/core')
import numpy as np
import cv2
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import matplotlib.pyplot as plt
from collections import OrderedDict
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from split_videos import save_frames
from DepthEstimation import get_device


device = get_device()

# get frames of video
frames_folder_path = "frames"
video_path = "frisbee_videos/red_tape_throws/gopro_frisbee_clip_8.mp4"
save_frames(video_path, frames_folder_path)



