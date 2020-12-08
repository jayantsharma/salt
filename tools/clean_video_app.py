import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Anonymize a batch of videos')
parser.add_argument('--resume', default='SiamMask_DAVIS.pth', type=str,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_dir', default='../../data/tennis', help='datasets')
args = parser.parse_args()

if __name__ == '__main__':
    base_command = f"python ../../tools/video_cleaner.py --resume {args.resume} --config {args.config}"
    for subfile in os.listdir(args.base_dir):
        if "mp4" in subfile.lower():
            command = f"{base_command} --base_path '{args.base_dir}/{subfile}' --target_path '../../{subfile[:-4]}/'"
            print(command)
            os.system(command)
