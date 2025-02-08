from project_config.diastolic_frame_config import get_args
from utils import superpoint
from utils import diastolic_frame
import os
import numpy as np
import cv2
import random


def get_diastolic_frame():
    args = get_args()
    subfolders = []
    for f in os.scandir(args.input):
        if f.is_dir():  # First-level subfolders
            for sub_f in os.scandir(f.path):
                if sub_f.is_dir():  # Second-level subfolders
                    subfolders.append(sub_f.path)

    random_frames = []  # Used to store all diastolic frame information
    for subfolder in subfolders:
        images = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]  # Get files in the folder
        random_image = random.choice(images)  # Choose a random image
        random_frames.append((subfolder, random_image))

    return random_frames


if __name__ == "__main__":
    random_frames = get_diastolic_frame()

    # Write the results to a txt file
    output_file = "data/random_frames.txt"
    with open(output_file, "w") as f:
        for subfolder, frame_name in random_frames:
            f.write(f"{subfolder}/{frame_name}\n")

    print(f"Results have been saved to {output_file}")
