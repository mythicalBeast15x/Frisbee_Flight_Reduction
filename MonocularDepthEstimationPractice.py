import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# looks for gpu then returns device or cpu if none is detected
def get_device():
    if torch.cuda.is_available():
        gpu_device = "cuda"
    elif torch.backends.mps.is_available():
        gpu_device = "mps"
    else:
        gpu_device = "cpu"
    print(f"Utilizing device: {gpu_device}")

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

cap = cv2.VideoCapture("frisbee_videos/red_tape_throws/gopro_frisbee_clip_8.mp4")

ret, first_frame = cap.read()

#for x in range(38):
#    ret, first_frame = cap.read()

#img = cv2.imread(first_frame)
if ret:
    img = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    output = (output*255).astype(np.uint8)
    output = cv2.applyColorMap(output, cv2.COLORMAP_MAGMA)

    cv2.imshow("Depth Map", output)

    cv2.waitKey(0)
    #if cv2.waitKey(5) & 0xFF == 27:
    #    break


    '''
    print(f'MAX: {np.max(output)}, MIN: {np.min(output)}, AVG: {np.mean(output)}')
    # setting definitions of depth
    desired_range = .40
    depth_range = np.max(output) - np.min(output)
    min_depth_of_disk = desired_range * depth_range
    print(f"min_depth_of_disk: {min_depth_of_disk }")

    indices = np.argwhere(output == np.max(output))
    '''

    """
    left_bound = 20
    right_bound = 30
    up_bound = 0
    down_bound = 0
    # look for left bound
    for x in range(indices[0][1]-1, -1, -1):
        #print(output[indices[0][0]][x])
        if output[indices[0][0]][x] < min_depth_of_disk or x == 0:
            left_bound = x
            break
    print((len(output[0]) - indices[0][1]))

    # right bound
    for x in range((len(output[0]) - indices[0][1])):
        print("lfrb",output[indices[0][0]][x+indices[0][1]])
        right_bound = output[indices[0][0]][x+indices[0][1]]
        if output[indices[0][0]][x+indices[0][1]] < min_depth_of_disk or x == (len(output[0]) - indices[0][1]) -1:
            right_bound = x+indices[0][1]
            break

    # up bound
    for x in range(indices[0][0]-1, -1, -1):
        print("ub",x)
        if output[x][indices[0][0]] < min_depth_of_disk or x == 0:
            up_bound = x
            break

    print(len(output), indices[0][0], (len(output) - indices[0][0]))
    # down bound
    for x in range((len(output) - indices[0][0])):
        print("db",x, x+indices[0][0])
        if output[x+indices[0][0]][indices[0][0]] < min_depth_of_disk or x == (len(output) - indices[0][0])-1:
            print(f"min: {min_depth_of_disk}, down bound: {output[x+indices[0][0]][indices[0][0]]}")
            down_bound = x+indices[0][0]
            break

    print("indicies",indices[0][0], indices[0][1])
    """
    #plt.imshow(output)
    #plt.plot(indices[0][1], indices[0][0], 'ro')

    """
    plt.plot(right_bound, indices[0][0], 'bo')
    print(right_bound)
    plt.plot(left_bound, indices[0][0], 'bo')
    plt.plot(indices[0][1], up_bound, 'bo')
    print(right_bound)
    print(f"up: {up_bound}\ndown: {down_bound}\nleft: {left_bound}\nright: {right_bound}")
    plt.plot(indices[0][1], down_bound, 'yo')
    
    """
    #plt.show()
    #while True:
        #ret, frame = cap.read()
else:
    print('no frame detected')
