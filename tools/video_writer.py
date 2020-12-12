# --------------------------------------------------------
# Anonymal
# Licensed under The MIT License
# Written by Eric Zelikman and Xindi Wu
# Based on SiamMask
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from test import *
from tqdm import tqdm
import os
import time

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--base_path', default='../../data/tennis', help='original video location')
parser.add_argument('--target_path', default='../../results/', help='save target folder path')
parser.add_argument('--video_name', default='result.mp4', help='resulting video name')
args = parser.parse_args()

if __name__ == '__main__':
    cap = cv2.VideoCapture(args.base_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.target_path + args.video_name, fourcc, fps, (width, height), True)
    f = 0
    start_time = time.time()
    ret, im = cap.read()
    while ret:
        if f % (fps) == 0: print(f'{f // fps} / {length // fps}')
        if str(f) + ".npz" in os.listdir(args.target_path):
            mask = np.load(args.target_path + str(f) + ".npz")["arr_0"]
            blur = cv2.blur(im, (30, 30), cv2.BORDER_DEFAULT)
            im = (mask == 0)[:, :, None] * blur + (mask != 0)[:, :, None] * im
        out.write(im)
        ret, im = cap.read()
        f += 1
    cap.release()
    out.release()
    print('Anonymal Time: {:02.1f}s'.format(time.time() - start_time))
