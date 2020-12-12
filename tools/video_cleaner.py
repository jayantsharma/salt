# --------------------------------------------------------
# Anonymal
# Licensed under The MIT License
# Written by Eric Zelikman and Xindi Wu
# Based on SiamMask
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import os
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

    def set_frame(self, new_f):
        if self.manual_update:
            self.f = min(max(new_f, 0), self.length)
            self.cap.set(1, self.f)
    
    def shift_frame(self, delta):
        self.set_frame(self.f + delta)
    
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

if __name__ == '__main__':
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    write_executor = Executor(max_workers=args.writers)
    window_name = "Anonymal"
    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    def get_siammask():
        siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            siammask = load_pretrain(siammask, args.resume)
        siammask.eval().to(device)
        return siammask

    base_siammask = get_siammask()

    target_value = 0
    toc = 0
    vid = VideoPlayer()
    ret, im = vid.get_cur_frame()

    key = cv2.waitKey(0)
    vid.mask_enabled = key != 99
    vid.get_new_example = vid.mask_enabled
    states = []
    metadata = {}
    cur_identity = get_max_identity(args.metadata_path)

    while ret:
        tic = cv2.getTickCount()
        # If in selection mode
        if vid.get_new_example and vid.mask_enabled:  # init
            cur_identity += len(states) # Count the previous boxes
            states = []
            init_rects = cv2.selectROIs(window_name, im, False, False)
            # Create a new siammask tracker for each box
            for rect_idx, init_rect in enumerate(init_rects):
                if sum(init_rect) != 0:
                    # We initiate new models judiciously, as most anonymization instances will only require one box
                    siammask = base_siammask if rect_idx == 0 else get_siammask()
                    x, y, w, h = init_rect
                    target_pos = np.array([x + w / 2, y + h / 2])
                    target_sz = np.array([w, h])
                    states.append(siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device))  # init tracker
                vid.get_new_example = False
            
            key = cv2.waitKey(0)
            print(key)
            if key == 97: vid.shift_frame(-12)
            elif key == 100: vid.shift_frame(12)
            elif key == 115: vid.mask_enabled = False
        else: # tracking
            # If you're actually tracking the objects
            if vid.mask_enabled:
                # Initiate the frame metadata
                metadata[vid.f] = {'index': vid.f, 'tracked_objects': []}
                all_mask = np.ones_like(im)[:, :, 0]
                for state_idx, state in enumerate(states):
                    states[state_idx] = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
                    # Add the object metadata to the list of tracked objects for this frame
                    metadata[vid.f]['tracked_objects'].append({
                        "boundingBox": state['ploygon'].tolist(),
                        "identity": cur_identity + state_idx, # Keep a different identity for each mask
                        "score": state["score"].item()
                    })
                    location = state['ploygon'].flatten()
                    mask = state['mask'] < state['p'].seg_thr
                    all_mask *= mask
                blur = cv2.blur(im, (30, 30), cv2.BORDER_DEFAULT)
                im = (all_mask == 0)[:, :, None] * blur + (all_mask != 0)[:, :, None] * im
                write_executor.submit(np.savez_compressed, args.target_path + str(vid.f) + ".npz", all_mask > 0)
            cv2.imshow(window_name, im)
            key = cv2.waitKey(1)
            if key > 0:
                if key == 97: vid.shift_frame(-48)
                elif key == 100: vid.shift_frame(48)
                else:
                    vid.mask_enabled = key != 99
                    vid.get_new_example = vid.mask_enabled
                
        toc += cv2.getTickCount() - tic
        ret, im = vid.next_frame()
    if args.metadata_path is not None:
        write_metadata(args.metadata_path, args.base_path, metadata)
    toc /= cv2.getTickFrequency()
    fps = vid.length / toc
    vid.end()
    print(f'{window_name} Time: {toc:02.1f}s Speed: {fps:3.1f}fps (with visulization!)')
    if confirm('Would you like to export the video? [Y/n] ', default=True):
        if os.path.exists(args.target_path + args.base_path.split("/")[-1]):
            if not confirm("WARNING: File already exists. Are you sure you want to overwrite the video? "):
                quit()
        print("Exporting video in the background!")
        subprocess.Popen(['python', '../../tools/video_writer.py', '--base_path', args.base_path, '--target_path', args.target_path, "--video_name", args.base_path.split("/")[-1]])
