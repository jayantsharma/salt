# --------------------------------------------------------
# Anonymal
# Licensed under The MIT License
# Written by Eric Zelikman and Xindi Wu
# Based on SiamMask
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import os
from copy import copy
import json
import subprocess
import time
from test import *
from concurrent.futures import ProcessPoolExecutor as Executor
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='base video path')
parser.add_argument('--target_path', default='../../results/', help='target path to store blurred images')
parser.add_argument('--writers', default=16, help='number of image writers')
parser.add_argument('--metadata_path', help='where to store (or append) the metadata')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

cfg = load_config(args)
from custom import Custom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def get_siammask():
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)
    siammask.eval().to(device)
    return siammask

def get_max_identity(metadata_path):
    cur_identity = 0
    # Get the max existing identity if adding to an existing file
    if metadata_path is not None and os.path.isfile(metadata_path):
        with open(metadata_path, "r") as cur_file:
            for frame in json.load(cur_file)["frames"]:
                for idx, item in frame.items():
                    if isinstance(item, list):
                        identities = [example["identity"] + 1 for example in item if "identity" in example]
                        cur_identity = max(cur_identity, *identities)
    return cur_identity

def write_metadata(metadata_path, base_path, metadata):
    # write metadata to file:
    old_metadata = {"filepath":base_path, "frames":[], "service":"blur", "out_type":"videos"}
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r") as cur_file:
            old_metadata = json.load(cur_file)
            for frame in old_metadata["frames"]:
                if frame["index"] in metadata:
                    frame['tracked_objects'] = metadata.pop(frame["index"])['tracked_objects']
    old_metadata["frames"].extend(metadata.values())
    with open(metadata_path, "w") as cur_file:
        json.dump(old_metadata, cur_file)

class VideoPlayer():
    def __init__(self):
        self.f = 0
        self.manual_update = True
        self.cap = cv2.VideoCapture(args.base_path)
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame = self.cap.read()
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.trackbar_name = "Current Frame"
        cv2.createTrackbar(self.trackbar_name, window_name, 0, self.length, self.set_frame)

        self.get_new_example = False
        self.mask_enabled = False

    def get_cur_frame(self):
        return self.frame

    def refresh_cur_frame(self):
        f = self.cap.get(1)
        if f > 0:
            f -= 1
            self.cap.set(1, f)
            self.frame = self.cap.read()
            return self.frame

    def set_frame(self, new_f):
        if self.manual_update:
            f = min(max(new_f, 0), self.length)
            self.cap.set(1, f)
            self.f = f - 1
    
    def shift_frame(self, delta):
        self.set_frame(self.f + delta)

    def shift_and_read_frame(self, delta):
        self.shift_frame(delta)
        return self.next_frame()
    
    def next_frame(self):
        self.f += 1
        self.manual_update = False
        cv2.setTrackbarPos(self.trackbar_name, window_name, self.f)
        self.manual_update = True
        self.frame = self.cap.read()
        return self.frame

    def end(self):
        self.cap.release()
        cv2.destroyWindow(window_name)

    def enter_mask_mode(self):
        self.mask_enabled = True
        self.get_new_example = True

def confirm(prompt, default=None):
    while True:
        answer = input(prompt)
        if len(answer) == 0 and default is not None:
            return default
        else:
            try:
                return bool(strtobool(answer))
            except:
                print("Please choose a valid option")
                continue

class Track():
    def __init__(self, offset, init_rect):
        self.offset = offset

        self.siammask = get_siammask()
        x, y, w, h = init_rect
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        state = siamese_init(im, target_pos, target_sz, self.siammask, cfg['hp'], device=device)  # init tracker
        self.state = state

        tracked_init = siamese_track(self.state, im, mask_enable=True, refine_enable=True, device=device)  # track
        rect = tracked_init['ploygon'] # top-left & bottom-right, xy format
        # rect = np.array([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])
        self.frames_tracked = [rect]

    def track_frame(self, im, f):
        if self.is_next_frame(f):
            tracked_state = siamese_track(self.state, im, mask_enable=True, refine_enable=True, device=device)  # track
            rect = tracked_state['ploygon'] # top-left & bottom-right, xy format
            self.frames_tracked.append(rect)
            self.state = tracked_state
            return rect
    
    def is_next_frame(self, f):
        return self.map_to_idx(f) == len(self.frames_tracked)

    def map_to_idx(self, f):
        return f - self.offset

    def map_to_f(self, idx):
        return self.offset + idx

    def get_rect(self, f):
        return self.frames_tracked[self.map_to_idx(f)]

    def terminate(self, f, per_frame_tracks):
        last_f = self.map_to_f(len(self.frames_tracked) - 1)
        for frm in range(f, last_f + 1):
            per_frame_tracks[frm].remove(self)
        self.frames_tracked = self.frames_tracked[:self.map_to_idx(f)]
        # Free up memory
        del self.siammask
        del self.state

if __name__ == '__main__':
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)

    # write_executor = Executor(max_workers=args.writers)
    window_name = "Anonymal"

    # base_siammask = get_siammask()

    target_value = 0
    vid = VideoPlayer()
    ret, im = vid.get_cur_frame()

    # key = cv2.waitKey(0)
    # vid.mask_enabled = key != 99
    # vid.get_new_example = vid.mask_enabled
    mode = "play"
    states = []
    metadata = {}
    cur_identity = get_max_identity(args.metadata_path)
    cur_tracks = set()
    per_frame_tracks = [ set() for _ in range(vid.length) ]

    def display(im, mode):
        if mode == "play":
            text = "{}, Frame {}, c: -30, v: +30, Space: Pause".format(mode.upper(), vid.f)
        elif mode == "review":
            text = "{}, Frame {}, d: -1, f: +1, c: -5, v: +5, m: Modify, n: New, Space: Play".format(mode.upper(), vid.f)
        elif mode == "roi":
            text = "Draw bbox, Frame {}".format(vid.f)
        elif mode == "modify":
            text = "Modify, Frame {}, t: Terminate, d: Delete".format(vid.f)
        cv2.putText(im, text, (100,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0))
        for track in per_frame_tracks[vid.f]:
            rect = track.get_rect(vid.f)
            clr = (255,0,0) if track in cur_tracks else (0,0,255) 
            # cv2.rectangle(im, tuple(rect[0]), tuple(rect[1]), clr, 2)
            cv2.polylines(im, [np.int0(rect).reshape((-1, 1, 2))], True, clr, 3)
        cv2.imshow(window_name, im)

    tic = time.time()
    while ret:
        if mode == "play":
            for track in cur_tracks:
                rect = track.track_frame(im, vid.f)
                if rect is not None:
                    per_frame_tracks[vid.f].add(track)
            display(im, mode)
            key = cv2.waitKey(1)
            if key > 0:
                if key == 99: 
                    ret, im = vid.shift_and_read_frame(-30)
                elif key == 118: 
                    vid.shift_and_read_frame(+30)
                elif key == 32:
                    mode = "review"
                    ret, im = vid.refresh_cur_frame()
            else:
                ret, im = vid.next_frame()

        elif mode == "review":
            display(im, mode)
            key = cv2.waitKey(0)
            if key > 0:
                if key in [100, 102, 99, 118]:
                    if key == 100:
                        d = -1
                    elif key == 102:
                        d = +1
                    elif key == 99:
                        d = -5
                    else:
                        d = +5
                    ret, im = vid.shift_and_read_frame(d)
                elif key == 109:
                    ret, im = vid.refresh_cur_frame()
                    tracks = copy(per_frame_tracks[vid.f])
                    for i, track in enumerate(tracks):
                        if track in cur_tracks:
                            text = "Track {}, Frame {}, t: Terminate, d: Delete, c: Continue".format(i + 1, vid.f)
                            clr = (255,0,0)
                        else:
                            text = "Track {}, Frame {}, d: Delete, c: Continue".format(i + 1, vid.f)
                            clr = (0,0,255)
                        cv2.putText(im, text, (100,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0))
                        rect = track.get_rect(vid.f)
                        cv2.polylines(im, [np.int0(rect).reshape((-1, 1, 2))], True, clr, 3)
                        cv2.imshow(window_name, im)
                        key = cv2.waitKey(0)
                        if track in cur_tracks:
                            if key in [116, 100]:
                                end_f = vid.f if key == 116 else track.offset
                                track.terminate(end_f, per_frame_tracks)
                                cur_tracks.remove(track)
                        else:
                            if key == 100:
                                track.terminate(track.offset, per_frame_tracks)
                        ret, im = vid.refresh_cur_frame()
                elif key == 110:
                    # select roi
                    ret, im = vid.refresh_cur_frame()
                    display(im, "roi")
                    init_rect = cv2.selectROI(window_name, im, False, False)
                    if init_rect:
                        track = Track(vid.f, init_rect)
                        cur_tracks.add(track)
                        per_frame_tracks[vid.f].add(track)

                    mode = "play"
                    ret, im = vid.next_frame()
                elif key == 32:
                    mode = "play"
                    ret, im = vid.refresh_cur_frame()
            else:
                continue

    # if args.metadata_path is not None:
    #     write_metadata(args.metadata_path, args.base_path, metadata)
    toc = time.time()
    tm = toc - tic
    fps = vid.length / tm
    vid.end()
    print(f'{window_name} Time: {tm:02.1f}s Speed: {fps:3.1f}fps (with visualization!)')
    # if confirm('Would you like to export the video? [Y/n] ', default=True):
    #     if os.path.exists(args.target_path + args.base_path.split("/")[-1]):
    #         if not confirm("WARNING: File already exists. Are you sure you want to overwrite the video? "):
    #             quit()
    #     print("Exporting video in the background!")
    #     subprocess.Popen(['python', '../../tools/video_writer.py', '--base_path', args.base_path, '--target_path', args.target_path, "--video_name", args.base_path.split("/")[-1]])
