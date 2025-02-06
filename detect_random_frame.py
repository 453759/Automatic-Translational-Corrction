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
        if f.is_dir():  # 一级子文件夹
            for sub_f in os.scandir(f.path):
                if sub_f.is_dir():  # 二级子文件夹
                    subfolders.append(sub_f.path)

    random_frames = []  # 用于存储所有舒张帧信息
    for subfolder in subfolders:
        images = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]  # 获取文件夹中的文件
        random_image = random.choice(images)
        random_frames.append((subfolder, random_image))

    return random_frames


if __name__ == "__main__":
    random_frames = get_diastolic_frame()

    # 将结果写入 txt 文件
    output_file = "data/random_frames.txt"
    with open(output_file, "w") as f:
        for subfolder, frame_name in random_frames:
            f.write(f"{subfolder}/{frame_name}\n")

    print(f"Results have been saved to {output_file}")
