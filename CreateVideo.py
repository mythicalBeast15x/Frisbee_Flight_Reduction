import cv2
import os

directory = "Notebooks/predicted_flows"
raw_imgs = [img for img in os.listdir(directory) if img.endswith(".jpg")]
imgs = ['' for _ in range(len(raw_imgs))]


for img in raw_imgs:
    index = int(img[15:-4])
    imgs[index] = img

if not imgs:
    raise ValueError("No images found in the directory.")

first_image_path = os.path.join(directory, imgs[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))

