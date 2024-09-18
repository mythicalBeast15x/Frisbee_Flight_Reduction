import cv2
import os
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def images_to_video(image_folder, output_video, fps=30, filetype=".jpg"):
    # Get list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(filetype)]
    images.sort(key=natural_sort_key)  # Ensure images are in the correct order

    # Read the first image to get dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the VideoWriter object
    video.release()
    print(f"Video saved as {output_video}")


"""
image_folder = 'Notebooks/predicted_flows'
output_video = 'clip14RAFT2.mp4'
images_to_video(image_folder, output_video)"""
