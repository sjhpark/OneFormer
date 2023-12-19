import cv2
import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='path to the video to be converted to frames')
    parser.add_argument('--fps', type=int, default=None, help='fps of the video')
    parser.add_argument('--save_dir', type=str, default="images_samples", help='directory to save the frames')
    args = parser.parse_args()

    assert args.path is not None, "Please provide the path to the video to be converted to frames"

    # save dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # video
    video = cv2.VideoCapture(args.path)

    # original fps of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Original FPS of the video: {fps}")

    # total frames in the video
    frames_tot = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {frames_tot}")

    # desired fps of the video
    fps_desired = args.fps
    print(f"Desired FPS of the video: {fps_desired}")

    # desired frames in the video
    frames_desired = round(frames_tot * fps_desired / fps)
    print(f"Desired frames in the video: {frames_desired}")

    # convert to & save frames
    success = True
    count = 0
    # while success:
    for i in tqdm(range(frames_tot), desc="Converting video to frames"):
        success, image = video.read()
        if success and count % int(fps / fps_desired) == 0:
            cv2.imwrite(os.path.join(save_dir, "frame%d.jpg" % count), image)  # save frame as JPEG file
        count += 1
    print(f"Video-to-Frames conversion is complete. All the frames are saved in {args.save_dir}")
    print(f"Total frames captured: {len(os.listdir(args.save_dir))}")
