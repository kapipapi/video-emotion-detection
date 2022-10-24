import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=(720, 1280), device=device)
mtcnn.to(device)

save_frames = 15
input_fps = 30

save_length = 3.6  # seconds

failed_videos = []
root = '../dataset/'
avoid_actor_number = list(range(1, 17 + 1))  # from 1 to 17

select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
for directory in sorted(os.listdir(root)):
    actor_number = int(directory.split("_")[1])
    if avoid_actor_number is not None and actor_number in avoid_actor_number:
        print(f"Avoiding actor with number {actor_number}")
        continue
    for filename in tqdm(os.listdir(os.path.join(root, directory))):
        if filename.endswith('.mp4'):
            cap = cv2.VideoCapture(os.path.join(root, directory, filename))

            # calculate length in frames
            framen = 0
            while True:
                i, q = cap.read()
                if not i:
                    break
                framen += 1
            cap = cv2.VideoCapture(os.path.join(root, directory, filename))

            if save_length * input_fps > framen:
                skip_begin = int((framen - (save_length * input_fps)) // 2)
                for i in range(skip_begin):
                    _, im = cap.read()

            framen = int(save_length * input_fps)
            frames_to_select = select_distributed(save_frames, framen)
            save_fps = save_frames // (framen // input_fps)

            numpy_video = []
            success = 0
            frame_ctr = 0

            while True:
                ret, im = cap.read()
                if not ret:
                    break
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except:
                    failed_videos.append((directory, i))
                    break

                temp = im[:, :, -1]
                im_rgb = im.copy()
                im_rgb[:, :, -1] = im_rgb[:, :, 0]
                im_rgb[:, :, 0] = temp
                im_rgb = torch.tensor(im_rgb)
                im_rgb = im_rgb.to(device)

                bbox = mtcnn.detect(im_rgb)
                if bbox[0] is not None:
                    bbox = bbox[0][0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                    im = im[y1:y2, x1:x2, :]
                    im = cv2.resize(im, (224, 224))
                    numpy_video.append(im)
            if len(frames_to_select) > 0:
                for i in range(len(frames_to_select)):
                    numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
            np.save(os.path.join(root, directory, filename[:-4] + '_facecroppad.npy'), np.array(numpy_video))
            if len(numpy_video) != 15:
                print('Error', directory, filename)
