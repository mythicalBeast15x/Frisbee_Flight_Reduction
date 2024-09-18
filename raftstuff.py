import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = "tight"
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
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
_ = urlretrieve(video_url, video_path)
from torchvision.io import read_video
frames, _, _ = read_video(str(video_path), output_format="TCHW")

img1_batch = torch.stack([frames[100], frames[150]])
img2_batch = torch.stack([frames[101], frames[151]])
plot(img1_batch)
from torchvision.models.optical_flow import Raft_Large_Weights
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()

def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 920], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 920], antialias=False)
    return transforms(img1_batch, img2_batch)

img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
print(f"shape = {img1_batch.shape}, dtype ={img2_batch.dtype}")
from torchvision.models.optical_flow import raft_large
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print(device)

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()
list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = num of itererations")
from torchvision.io import write_jpeg
from torchvision.utils import flow_to_image
#model.eval()
for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
    img1, img2 = preprocess(img1.unsqueeze(0), img2.unsqueeze(0))
    list_of_flows = model(img1.to(device), img2.to(device))
    predicted_flow = list_of_flows[-1][0]
    flow_img = flow_to_image(predicted_flow).to("cpu")
    output_folder = "Notebooks/predicted_flows/"
    write_jpeg(flow_img, output_folder+f"predicted_flow_{i}.jpg")