import cv2
import os


def save_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = os.path.join(output_folder, f'{video_path[-11:-4]}frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        print(frame_filename, "written")
        frame_count += 1
    cap.release()


def get_filenames(folders):
    filenames = []
    for folder in folders:
        try:
            for file in os.listdir(folder):
                filenames.append(os.path.join(folder, file))
        except FileNotFoundError:
            print(f"Error: The folder {folder} does not exist.")
            return []

    return filenames


output_destination = 'flying_disc_dataset/video_frames'
input_folder_paths = ["frisbee_videos/red_tape_throws"]     # Enter folder pathways here
filenames = get_filenames(input_folder_paths)
for file in filenames:
    print(file)
    save_frames(file, output_destination)
