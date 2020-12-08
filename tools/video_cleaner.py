# --------------------------------------------------------
# Anonymal
# Licensed under The MIT License
# Written by Eric Zelikman and Xindi Wu
# Based on SiamMask
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from test import *
import os
from concurrent.futures import ProcessPoolExecutor as Executor

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--target_path', default='../../results/', help='datasets')
parser.add_argument('--writers', default=16, help='number of image writers')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

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

    # Parse Image file
    cap = cv2.VideoCapture(args.base_path)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret = True
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    toc = 0

    ret, im = cap.read()
    f = 0
    framerate = 60
    key = cv2.waitKey(0)
    mask_enabled = key != 99
    get_new_example = mask_enabled
    states = []
    while ret:
        tic = cv2.getTickCount()
        # If in selection mode
        if get_new_example and mask_enabled:  # init
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
                get_new_example = False
            
            key = cv2.waitKey(0)
            print(key)
            if key == 97:
                f -= 12
                cap.set(1, f)
            elif key == 100:
                f += 12
                cap.set(1, f)
            elif key == 115:
                mask_enabled = False
        elif f > 0:  # tracking
            # If you're actually tracking the objects
            if mask_enabled:
                for state_idx, state in enumerate(states):
                    states[state_idx] = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
                    location = state['ploygon'].flatten()
                    mask = state['mask'] > state['p'].seg_thr * 0.9
                    
                    blur = cv2.blur(im, (30, 30), cv2.BORDER_DEFAULT)
                    # im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                    im = (mask > 0)[:, :, None] * blur + (mask == 0)[:, :, None] * im
                    # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                    
                    write_executor.submit(cv2.imwrite, args.target_path + str(f) + ".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imshow(window_name, im)
            key = cv2.waitKey(1)
            if key > 0:
                print(key)
                if key == 97:
                    f -= 48
                    cap.set(1, f)
                elif key == 100:
                    f += 48
                    cap.set(1, f)
                else:
                    mask_enabled = key != 99
                    get_new_example = mask_enabled
                
        toc += cv2.getTickCount() - tic
        ret, im = cap.read()
        f += 1
    toc /= cv2.getTickFrequency()
    fps = f / toc
    cap.release()
    print(window_name + ' Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
