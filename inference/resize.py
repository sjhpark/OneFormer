import os
import cv2
import time
import argparse
from tqdm import tqdm
from natsort import natsorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None, help='Directory of where the input images are')
    parser.add_argument('--scale', nargs='+', type=float, default=[0.5,0.5], help='fx and fy for resizing')
    args = parser.parse_args()

    assert args.in_dir is not None, "Please provide the path to the video to be converted to images"
    assert len(args.scale) == 2, "Please provide 2 inputs (fx and fy) as sacle factors used for resizing"

    images = natsorted(os.listdir(args.in_dir))
    start = time.time()
    for img_name in tqdm(images, desc="Resizing images..."):
        if not os.path.exists(f'{args.in_dir}/resized'):
            os.makedirs(f'{args.in_dir}/resized')
        img = f'{args.in_dir}/{img_name}'
        img = cv2.imread(img)
        out = cv2.resize(img, (0,0), fx=args.scale[0], fy=args.scale[1], interpolation=cv2.INTER_NEAREST)
        out_name = f'{args.in_dir}/resized/{img_name[:-4]}_resize.png'
        cv2.imwrite(out_name, out)
    print(f"Resizing images is complete. All the images are saved in {args.in_dir}/resized")
    print(f"Total time elapsed: {(time.time() - start):.2f}s")

