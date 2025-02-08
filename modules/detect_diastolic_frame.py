from project_config.diastolic_frame_config import get_args
from utils import superpoint
from utils import diastolic_frame
import os
import numpy as np
import cv2


def get_diastolic_frame():
    args = get_args()
    subfolders = []
    for f in os.scandir(args.input):
        if f.is_dir():  # First-level subfolders
            for sub_f in os.scandir(f.path):
                if sub_f.is_dir():  # Second-level subfolders
                    subfolders.append(sub_f.path)

    diastolic_frames = []  # Used to store all diastolic frame information
    for subfolder in subfolders:
        vs = superpoint.VideoStreamer(subfolder, args.camid, args.H, args.W, args.skip, args.img_glob)
        fe = superpoint.SuperPointFrontend(weights_path=args.weights_path,
                                           nms_dist=args.nms_dist,
                                           conf_thresh=args.conf_thresh,
                                           nn_thresh=args.nn_thresh,
                                           cuda=args.cuda)
        tracker = superpoint.PointTracker(args.max_length, nn_thresh=fe.nn_thresh)

        # Initialize a dictionary to store image names and their corresponding points
        img_pts_dict = {}
        # Store the points of each frame in the dictionary
        while True:
            # Get a new image.
            img, status, img_name = vs.next_frame()
            if status is False:
                break

            pts, desc, heatmap = fe.run(img)
            if img_name is not None:
                img_pts_dict[img_name] = pts[:2, :].T

            # Add points and descriptors to the tracker.
            tracker.update(pts, desc)

        # Detect diastolic frames
        diastolic_frame_detector = diastolic_frame.DiastolicFrameDetector(img_pts_dict)
        diastolic_frame_name = diastolic_frame_detector.diastolic_frame
        print(diastolic_frame_name)
        diastolic_frames.append((subfolder, diastolic_frame_name))

    return diastolic_frames


if __name__ == "__main__":
    diastolic_frames = get_diastolic_frame()

    # Write results to a txt file
    output_file = "data/diastolic_frames_R.txt"
    with open(output_file, "w") as f:
        for subfolder, frame_name in diastolic_frames:
            print(f'subfolder={subfolder}')
            f.write(f"{subfolder}/{frame_name}\n")

    print(f"Results have been saved to {output_file}")
