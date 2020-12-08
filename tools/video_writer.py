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
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--target_path', default='../../results/', help='datasets')
args = parser.parse_args()

if __name__ == '__main__':
    cap = cv2.VideoCapture(args.base_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.target_path + "result.mp4", fourcc, fps, (width, height))
    f = 0
    start_time = time.time()
    ret, im = cap.read()
    while ret:
        if f % (fps) == 0: print(f // fps)
        if str(f) + ".png" in os.listdir(args.target_path):
            im = cv2.imread(args.target_path + str(f) + ".png")
        out.write(im)
        ret, im = cap.read()
        f += 1
    total_time = int((time.time() - start_time) // 1000)
    cap.release()
    out.release()
    print('Anonymal Time: {:02.1f}s'.format(total_time))
